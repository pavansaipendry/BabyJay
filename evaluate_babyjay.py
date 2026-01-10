"""
BabyJay Evaluation Framework
============================
Tests retrieval accuracy, answer quality, and end-to-end performance

Metrics:
- Retrieval: Precision, Recall, MRR (Mean Reciprocal Rank)
- Answer Quality: Correctness, Relevance, Hallucination Detection
- End-to-End: Overall accuracy

Usage:
    python evaluate_babyjay.py
"""

import json
from typing import List, Dict, Tuple
from app.rag.retriever import Retriever
from app.rag.chat import BabyJayChat


class BabyJayEvaluator:
    """Evaluate BabyJay's retrieval and response quality."""
    
    def __init__(self):
        self.retriever = Retriever()
        self.chat = BabyJayChat(use_redis=False)  # Don't save test conversations
    
    # ========== TEST DATASETS ==========
    
    def get_faculty_test_cases(self) -> List[Dict]:
        """
        Test cases for faculty search.
        Each case has: query, expected_professors, expected_departments
        """
        return [
            {
                "query": "Find ML professors",
                "expected_names": ["Jian Li", "Karthik Srinivasan", "Amirmasoud", "Michael Branicky"],
                "expected_departments": ["EECS", "Business"],
                "category": "abbreviation"
            },
            {
                "query": "physics department professors who are interested in ML",
                "expected_names": ["Kyoungchul", "Kong", "Elliot", "Reynolds", "Brunetti"],
                "expected_departments": ["Physics"],
                "category": "department_filter"
            },
            {
                "query": "EECS professors doing robotics",
                "expected_names": ["Arvin Agah", "David Johnson", "Michael Branicky"],
                "expected_departments": ["EECS"],
                "category": "department_topic"
            },
            {
                "query": "deep learning researchers",
                "expected_names": ["Fengjun Li", "Elliot", "Reynolds"],
                "expected_departments": ["EECS", "Physics"],
                "category": "research_area"
            },
            {
                "query": "AI professors",
                "expected_names": ["Arvin Agah", "Karthik Srinivasan"],
                "expected_departments": ["EECS", "Business"],
                "category": "abbreviation"
            },
            {
                "query": "quantum computing researchers",
                "expected_names": ["Kyoungchul", "Kong", "Esam El-Araby"],
                "expected_departments": ["Physics", "EECS"],
                "category": "research_area"
            },
            {
                "query": "Business school machine learning",
                "expected_names": ["Karthik Srinivasan"],
                "expected_departments": ["Business"],
                "category": "department_topic"
            },
        ]
    
    def get_department_filter_test_cases(self) -> List[Dict]:
        """Test cases for department filtering follow-ups."""
        return [
            {
                "initial_query": "Find ML professors",
                "filter_query": "EECS only",
                "expected_departments": ["EECS"],
                "excluded_departments": ["Business", "Physics"],
                "category": "filter_eecs"
            },
            {
                "initial_query": "Show me AI faculty",
                "filter_query": "Just Business",
                "expected_departments": ["Business"],
                "excluded_departments": ["EECS", "Physics"],
                "category": "filter_business"
            },
        ]
    
    def get_general_test_cases(self) -> List[Dict]:
        """Test cases for non-faculty queries."""
        return [
            {
                "query": "Where can I eat on campus?",
                "expected_keywords": ["dining", "North College", "Mrs. E's"],
                "category": "dining"
            },
            {
                "query": "How much is tuition?",
                "expected_keywords": ["376.60", "11,298", "resident"],
                "category": "tuition"
            },
            {
                "query": "What bus goes to engineering?",
                "expected_keywords": ["bus", "route", "engineering"],
                "category": "transit"
            },
        ]
    
    # ========== RETRIEVAL METRICS ==========
    
    def calculate_retrieval_precision(self, retrieved: List[str], expected: List[str]) -> float:
        """
        Precision = (relevant retrieved) / (total retrieved)
        How many of the retrieved items are relevant?
        """
        if not retrieved:
            return 0.0
        
        relevant_retrieved = sum(1 for name in retrieved if any(exp.lower() in name.lower() for exp in expected))
        return relevant_retrieved / len(retrieved)
    
    def calculate_retrieval_recall(self, retrieved: List[str], expected: List[str]) -> float:
        """
        Recall = (relevant retrieved) / (total relevant)
        How many of the relevant items did we retrieve?
        """
        if not expected:
            return 1.0
        
        relevant_retrieved = sum(1 for exp in expected if any(exp.lower() in name.lower() for name in retrieved))
        return relevant_retrieved / len(expected)
    
    def calculate_mrr(self, retrieved: List[str], expected: List[str]) -> float:
        """
        Mean Reciprocal Rank - position of first relevant result
        1.0 if first result is relevant, 0.5 if second, 0.33 if third, etc.
        """
        for rank, name in enumerate(retrieved, 1):
            if any(exp.lower() in name.lower() for exp in expected):
                return 1.0 / rank
        return 0.0
    
    # ========== TEST EXECUTION ==========
    
    def test_faculty_retrieval(self) -> Dict:
        """Test faculty search retrieval accuracy."""
        print("\n" + "="*80)
        print("FACULTY RETRIEVAL TESTS")
        print("="*80)
        
        test_cases = self.get_faculty_test_cases()
        results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest {i}/{len(test_cases)}: {case['query']}")
            print("-" * 80)
            
            # Run retrieval
            search_results = self.retriever.smart_search(case['query'], n_results=5)
            faculty_results = search_results.get('faculty', [])
            
            # Extract retrieved names
            retrieved_names = [
                r.get('metadata', {}).get('name', '')
                for r in faculty_results
            ]
            
            # Extract retrieved departments
            retrieved_depts = [
                r.get('metadata', {}).get('department', '')
                for r in faculty_results
            ]
            
            # Calculate metrics
            precision = self.calculate_retrieval_precision(retrieved_names, case['expected_names'])
            recall = self.calculate_retrieval_recall(retrieved_names, case['expected_names'])
            mrr = self.calculate_mrr(retrieved_names, case['expected_names'])
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Check department accuracy
            dept_correct = any(
                exp_dept.lower() in dept.lower() 
                for dept in retrieved_depts 
                for exp_dept in case['expected_departments']
            ) if retrieved_depts else False
            
            # Print results
            print(f"Retrieved: {retrieved_names[:3]}")
            print(f"Expected: {case['expected_names'][:3]}")
            print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | MRR: {mrr:.2f}")
            print(f"Department Match: {'✓' if dept_correct else '✗'}")
            
            results.append({
                "query": case['query'],
                "category": case['category'],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mrr": mrr,
                "dept_correct": dept_correct,
                "retrieved_count": len(retrieved_names)
            })
        
        # Aggregate metrics
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        avg_mrr = sum(r['mrr'] for r in results) / len(results)
        dept_accuracy = sum(1 for r in results if r['dept_correct']) / len(results)
        
        print("\n" + "="*80)
        print("FACULTY RETRIEVAL SUMMARY")
        print("="*80)
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall: {avg_recall:.2f}")
        print(f"Average F1 Score: {avg_f1:.2f}")
        print(f"Average MRR: {avg_mrr:.2f}")
        print(f"Department Accuracy: {dept_accuracy:.2f}")
        
        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_mrr": avg_mrr,
            "dept_accuracy": dept_accuracy,
            "detailed_results": results
        }
    
    def test_department_filtering(self) -> Dict:
        """Test department filtering accuracy."""
        print("\n" + "="*80)
        print("DEPARTMENT FILTERING TESTS")
        print("="*80)
        
        test_cases = self.get_department_filter_test_cases()
        results = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest {i}/{len(test_cases)}: {case['initial_query']} → {case['filter_query']}")
            print("-" * 80)
            
            # Create new chat for each test
            chat = BabyJayChat(use_redis=False)
            
            # Initial query
            chat.ask(case['initial_query'])
            
            # Filter query
            response = chat.ask(case['filter_query'])
            
            # Get last retrieval results
            search_results = self.retriever.smart_search(
                f"{case['initial_query']} {case['filter_query']}", 
                n_results=5
            )
            faculty_results = search_results.get('faculty', [])
            
            # Extract departments
            retrieved_depts = [
                r.get('metadata', {}).get('department', '')
                for r in faculty_results
            ]
            
            # Check if ONLY expected departments present
            only_expected = all(
                any(exp.lower() in dept.lower() for exp in case['expected_departments'])
                for dept in retrieved_depts
            ) if retrieved_depts else False
            
            # Check if excluded departments are absent
            no_excluded = not any(
                any(excl.lower() in dept.lower() for excl in case['excluded_departments'])
                for dept in retrieved_depts
            ) if retrieved_depts else True
            
            filter_works = only_expected and no_excluded
            
            print(f"Retrieved Departments: {retrieved_depts}")
            print(f"Expected: {case['expected_departments']}")
            print(f"Excluded: {case['excluded_departments']}")
            print(f"Filter Correct: {'✓' if filter_works else '✗'}")
            
            results.append({
                "initial_query": case['initial_query'],
                "filter_query": case['filter_query'],
                "filter_correct": filter_works,
                "only_expected": only_expected,
                "no_excluded": no_excluded
            })
        
        # Calculate accuracy
        accuracy = sum(1 for r in results if r['filter_correct']) / len(results)
        
        print("\n" + "="*80)
        print("DEPARTMENT FILTERING SUMMARY")
        print("="*80)
        print(f"Filter Accuracy: {accuracy:.2f}")
        
        return {
            "accuracy": accuracy,
            "detailed_results": results
        }
    
    def test_answer_quality(self) -> Dict:
        """Test end-to-end answer quality."""
        print("\n" + "="*80)
        print("ANSWER QUALITY TESTS")
        print("="*80)
        
        faculty_cases = self.get_faculty_test_cases()
        general_cases = self.get_general_test_cases()
        all_cases = faculty_cases + general_cases
        
        results = []
        
        for i, case in enumerate(all_cases, 1):
            print(f"\nTest {i}/{len(all_cases)}: {case['query']}")
            print("-" * 80)
            
            # Get answer
            chat = BabyJayChat(use_redis=False)
            response = chat.ask(case['query'])
            
            # Check for expected content
            has_expected = False
            if 'expected_names' in case:
                # Faculty query
                has_expected = any(name.lower() in response.lower() for name in case['expected_names'])
            elif 'expected_keywords' in case:
                # General query
                has_expected = any(kw.lower() in response.lower() for kw in case['expected_keywords'])
            
            # Check for hallucination indicators
            hallucination_flags = [
                "i don't have" in response.lower() and has_expected,  # Says no info but we have it
                "check the website" in response.lower() and has_expected,  # Deflects but we have it
            ]
            has_hallucination = any(hallucination_flags)
            
            # Check length appropriateness
            appropriate_length = 50 < len(response) < 2000
            
            print(f"Response Length: {len(response)} chars")
            print(f"Contains Expected: {'✓' if has_expected else '✗'}")
            print(f"Hallucination: {'✗' if has_hallucination else '✓'}")
            print(f"Appropriate Length: {'✓' if appropriate_length else '✗'}")
            
            results.append({
                "query": case['query'],
                "has_expected": has_expected,
                "no_hallucination": not has_hallucination,
                "appropriate_length": appropriate_length,
                "response_length": len(response)
            })
        
        # Calculate metrics
        accuracy = sum(1 for r in results if r['has_expected']) / len(results)
        hallucination_rate = sum(1 for r in results if not r['no_hallucination']) / len(results)
        length_appropriateness = sum(1 for r in results if r['appropriate_length']) / len(results)
        
        print("\n" + "="*80)
        print("ANSWER QUALITY SUMMARY")
        print("="*80)
        print(f"Answer Accuracy: {accuracy:.2f}")
        print(f"Hallucination Rate: {hallucination_rate:.2f}")
        print(f"Length Appropriateness: {length_appropriateness:.2f}")
        
        return {
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "length_appropriateness": length_appropriateness,
            "detailed_results": results
        }
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite."""
        print("\n" + "="*80)
        print("BABYJAY COMPREHENSIVE EVALUATION")
        print("="*80)
        
        # Run all tests
        retrieval_results = self.test_faculty_retrieval()
        filtering_results = self.test_department_filtering()
        answer_results = self.test_answer_quality()
        
        # Overall summary
        print("\n" + "="*80)
        print("OVERALL PERFORMANCE SUMMARY")
        print("="*80)
        print(f"\n📊 Retrieval Performance:")
        print(f"   Precision: {retrieval_results['avg_precision']:.2f}")
        print(f"   Recall: {retrieval_results['avg_recall']:.2f}")
        print(f"   F1 Score: {retrieval_results['avg_f1']:.2f}")
        print(f"   MRR: {retrieval_results['avg_mrr']:.2f}")
        
        print(f"\n🎯 Department Filtering:")
        print(f"   Accuracy: {filtering_results['accuracy']:.2f}")
        
        print(f"\n💬 Answer Quality:")
        print(f"   Accuracy: {answer_results['accuracy']:.2f}")
        print(f"   Hallucination Rate: {answer_results['hallucination_rate']:.2f}")
        
        # Overall grade
        overall_score = (
            retrieval_results['avg_f1'] * 0.4 +
            filtering_results['accuracy'] * 0.3 +
            answer_results['accuracy'] * 0.3
        )
        
        print(f"\n🏆 Overall Score: {overall_score:.2f}")
        
        if overall_score >= 0.9:
            grade = "A (Excellent)"
        elif overall_score >= 0.8:
            grade = "B (Good)"
        elif overall_score >= 0.7:
            grade = "C (Acceptable)"
        else:
            grade = "D (Needs Improvement)"
        
        print(f"   Grade: {grade}")
        
        return {
            "retrieval": retrieval_results,
            "filtering": filtering_results,
            "answer_quality": answer_results,
            "overall_score": overall_score,
            "grade": grade
        }
    
    def save_results(self, results: Dict, filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {filename}")


def main():
    """Run evaluation."""
    evaluator = BabyJayEvaluator()
    results = evaluator.run_full_evaluation()
    evaluator.save_results(results)


if __name__ == "__main__":
    main()