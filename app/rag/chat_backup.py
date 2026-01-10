"""
BabyJay Chat - CONSERVATIVE FIX
================================
Only fixes department filtering, keeps everything else the same.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict
from openai import OpenAI
from app.rag.retriever import Retriever

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# System prompt - slightly improved
SYSTEM_PROMPT = """You are BabyJay, KU's campus assistant. You help students find information about faculty, courses, dining, housing, transit, admissions, financial aid, tuition, campus buildings, student organizations, recreation, libraries, and safety.

CRITICAL RULES:
1. If context is provided, USE IT. Answer from the context.
2. If NO context is provided, say you don't have that information and suggest who to contact.
3. Be conversational - brief for simple questions, detailed for complex ones.
4. Never say "I don't have information" if context was actually provided.

IMPORTANT: When the user filters by department (e.g., "EECS only", "Just Business"), show ONLY professors from that exact department. Do not include professors from other departments even if they do similar research."""


class ConversationStore:
    """Handles persistent storage of conversations using Redis."""
    
    def __init__(self, redis_host='localhost', redis_port=6379, use_redis=True):
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.redis_client = None
        self.memory_store = {}
        
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
                print("✓ Connected to Redis for persistent conversation storage")
            except (redis.ConnectionError, redis.TimeoutError) as e:
                print(f"⚠️  Could not connect to Redis: {e}")
                print("   Falling back to in-memory storage")
                self.use_redis = False
    
    def save_message(self, session_id: str, role: str, content: str, ttl_days: int = 30):
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        
        if self.use_redis:
            key = f"chat:history:{session_id}"
            self.redis_client.rpush(key, json.dumps(message))
            self.redis_client.expire(key, 86400 * ttl_days)
        else:
            if session_id not in self.memory_store:
                self.memory_store[session_id] = []
            self.memory_store[session_id].append(message)
    
    def load_history(self, session_id: str, max_messages: int = 100) -> List[Dict]:
        if self.use_redis:
            key = f"chat:history:{session_id}"
            messages = self.redis_client.lrange(key, -max_messages, -1)
            return [json.loads(msg) for msg in messages]
        else:
            return self.memory_store.get(session_id, [])[-max_messages:]
    
    def clear_history(self, session_id: str):
        if self.use_redis:
            key = f"chat:history:{session_id}"
            self.redis_client.delete(key)
        else:
            if session_id in self.memory_store:
                del self.memory_store[session_id]
    
    def list_sessions(self, pattern: str = "chat:history:*") -> List[str]:
        if self.use_redis:
            keys = self.redis_client.keys(pattern)
            return [key.replace("chat:history:", "") for key in keys]
        else:
            return list(self.memory_store.keys())
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        history = self.load_history(session_id)
        if not history:
            return None
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "first_message": history[0]["timestamp"] if history else None,
            "last_message": history[-1]["timestamp"] if history else None,
            "preview": history[-1]["content"][:100] if history else None
        }


class BabyJayChat:
    def __init__(self, session_id: Optional[str] = None, use_redis: bool = True, debug: bool = False):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.retriever = Retriever()
        self.session_id = session_id or str(uuid.uuid4())
        self.store = ConversationStore(use_redis=use_redis)
        self.recent_context = ""
        self.last_search_query = ""
        self.debug = debug
        
        self._conversation_history: List[Dict] = []
        self._load_from_store()
    
    def _load_from_store(self):
        stored_history = self.store.load_history(self.session_id)
        self._conversation_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in stored_history
        ]
    
    def _save_message(self, role: str, content: str):
        self._conversation_history.append({"role": role, "content": content})
        self.store.save_message(self.session_id, role, content)
    
    @property
    def conversation_history(self) -> List[Dict]:
        return self._conversation_history
    
    def clear_history(self):
        self._conversation_history = []
        self.store.clear_history(self.session_id)
        self.recent_context = ""
        self.last_search_query = ""
        print(f"✓ Cleared conversation history for session {self.session_id[:8]}...")
    
    def switch_session(self, session_id: str):
        self.session_id = session_id
        self._load_from_store()
        self.recent_context = ""
        self.last_search_query = ""
        print(f"✓ Switched to session {session_id[:8]}...")
    
    def list_past_sessions(self) -> List[Dict]:
        session_ids = self.store.list_sessions()
        sessions = []
        
        for sid in session_ids:
            info = self.store.get_session_info(sid)
            if info:
                sessions.append(info)
        
        sessions.sort(key=lambda x: x["last_message"], reverse=True)
        return sessions
    
    def _is_simple_followup(self, question: str) -> bool:
        q = question.lower()
        
        if len(question.split()) > 8:
            return False
        
        simple_indicators = ['his', 'her', 'their', 'it', 'that', 'this', 'what about', 'how about']
        return any(ind in q for ind in simple_indicators)
    
    def _is_department_filter(self, question: str) -> bool:
        """Check if this is a department filtering request."""
        q = question.lower()
        
        # More lenient - just check for filter patterns and departments
        filter_patterns = ['only', 'just']
        dept_keywords = [
            'eecs', 'electrical', 'computer science', 'business', 'physics',
            'chemistry', 'math', 'engineering', 'department'
        ]
        
        has_filter = any(pattern in q for pattern in filter_patterns)
        has_dept = any(keyword in q for keyword in dept_keywords)
        
        return has_filter and has_dept
    
    def _extract_department(self, question: str) -> Optional[str]:
        """Extract department name - returns the keyword, not full name."""
        q = question.lower()
        
        # Return simple keywords that will be used in filtering
        if any(kw in q for kw in ["eecs", "computer science", "electrical engineering", " cs "]):
            return "Electrical Engineering and Computer Science"
        if "business" in q:
            return "School of Business"
        if "physics" in q:
            return "Department of Physics and Astronomy"
        if "chemistry" in q or " chem " in q:
            return "Department of Chemistry"
        if "math" in q:
            return "Department of Mathematics"
        if "psychology" in q or " psych " in q:
            return "Department of Psychology"
        if "mechanical" in q:
            return "Mechanical Engineering"
        
        return None
    
    def _filter_context_by_department(self, context: str, department: str) -> str:
        """Filter context to only include professors from specified department."""
        if not context or "=== FACULTY INFORMATION ===" not in context:
            return context
        
        lines = context.split('\n')
        filtered_lines = []
        include_current = False
        current_dept = ""
        buffer = []
        
        for line in lines:
            if line.strip() == "=== FACULTY INFORMATION ===":
                filtered_lines.append(line)
                include_current = False
                continue
            
            if line.strip().startswith("===") and "FACULTY" not in line:
                if include_current:
                    filtered_lines.extend(buffer)
                buffer = []
                include_current = False
                filtered_lines.append(line)
                continue
            
            if line.strip().startswith("Professor:"):
                if include_current:
                    filtered_lines.extend(buffer)
                buffer = [line]
                include_current = False
                current_dept = ""
                continue
            
            if line.strip().startswith("Department:"):
                current_dept = line.strip().replace("Department:", "").strip()
                buffer.append(line)
                # Check if this department matches our filter
                if department.lower() in current_dept.lower():
                    include_current = True
                continue
            
            buffer.append(line)
        
        # Flush final buffer
        if include_current:
            filtered_lines.extend(buffer)
        
        return '\n'.join(filtered_lines)
    
    def _expand_followup_question(self, question: str) -> str:
        if not self._conversation_history:
            return question
        
        recent_history = self._conversation_history[-6:]
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}"
            for msg in recent_history
        ])
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite the follow-up question to include full context. Return only the rewritten question."
                    },
                    {
                        "role": "user",
                        "content": f"Conversation:\n{history_text}\n\nFollow-up: {question}\n\nRewritten:"
                    }
                ],
                temperature=0,
                max_tokens=100
            )
            
            expanded = response.choices[0].message.content.strip()
            if expanded and len(expanded) > len(question):
                return expanded
            return question
            
        except Exception:
            return question
    
    def ask(self, question: str, use_history: bool = True) -> str:
        """Process a question and return a response."""
        
        # Department filter request
        if self._is_department_filter(question) and self.last_search_query:
            department = self._extract_department(question)
            
            if department:
                if self.debug:
                    print(f"[DEBUG] Department filter: {department}")
                
                # Re-run last search with department
                search_query = f"{self.last_search_query} {department}"
                results = self.retriever.smart_search(search_query, n_results=5)
                context = results.get("context", "")
                
                # STRICT filter
                context = self._filter_context_by_department(context, department)
                
                if self.debug:
                    print(f"[DEBUG] Filtered context: {len(context)} chars")
            else:
                context = ""
        else:
            # Normal search
            search_query = question
            
            if self._is_simple_followup(question) and use_history:
                expanded = self._expand_followup_question(question)
                if self.debug:
                    print(f"[DEBUG] Expanded: '{question}' → '{expanded}'")
                search_query = expanded
            
            # Track faculty queries
            if any(kw in question.lower() for kw in ['professor', 'faculty', 'research', 'ml', 'ai', 'deep learning', 'machine learning']):
                self.last_search_query = search_query
            
            # Search
            results = self.retriever.smart_search(search_query, n_results=5)
            context = results.get("context", "")
            
            if self.debug:
                print(f"[DEBUG] Query: '{search_query}'")
                print(f"[DEBUG] Context: {len(context)} chars")
            
            # Retry
            if not context:
                retry_query = f"KU {search_query}"
                if self.debug:
                    print(f"[DEBUG] Retry: '{retry_query}'")
                retry_results = self.retriever.smart_search(retry_query, n_results=5)
                context = retry_results.get("context", "")
            
            # Follow-up fallback
            if self._is_simple_followup(question) and self.recent_context and not context:
                context = self.recent_context
                if self.debug:
                    print(f"[DEBUG] Using recent_context")
        
        # Store context
        if context:
            self.recent_context = context
        
        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if use_history and self._conversation_history:
            messages.extend(self._conversation_history[-20:])
        
        if context:
            user_message = f"""I have this information from KU's database:

{context}

Question: {question}

Please answer using the information above."""
        else:
            user_message = f"""Question: {question}

I don't have specific information about this. Please suggest the user check official KU resources or contact the relevant department."""
        
        messages.append({"role": "user", "content": user_message})
        
        if self.debug:
            print(f"[DEBUG] Sending to LLM with context: {bool(context)}")
        
        # Get response
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            
            assistant_message = response.choices[0].message.content
            
            if use_history:
                self._save_message("user", question)
                self._save_message("assistant", assistant_message)
            
            return assistant_message
            
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}"


def main():
    """Interactive CLI."""
    print("=" * 60)
    print("🦅 BabyJay - Conservative Fix")
    print("=" * 60)
    
    if REDIS_AVAILABLE:
        print("✓ Redis enabled")
    else:
        print("⚠️  Redis unavailable")
    
    print("\nCommands: 'quit', 'clear', 'new', 'list', 'switch <id>', 'debug on/off'")
    print("-" * 60)
    
    chat = BabyJayChat()
    print(f"\nSession: {chat.session_id[:8]}...")
    
    if chat.conversation_history:
        print(f"Continuing ({len(chat.conversation_history)} messages)")
    else:
        print("New conversation")
    
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Rock Chalk! 🏀")
                break
            
            if user_input.lower() == 'clear':
                chat.clear_history()
                continue
            
            if user_input.lower() == 'new':
                chat = BabyJayChat()
                print(f"✓ New session: {chat.session_id[:8]}...")
                continue
            
            if user_input.lower() == 'debug on':
                chat.debug = True
                print("✓ Debug ON")
                continue
            
            if user_input.lower() == 'debug off':
                chat.debug = False
                print("✓ Debug OFF")
                continue
            
            if user_input.lower() == 'list':
                sessions = chat.list_past_sessions()
                if sessions:
                    print("\nPast conversations:")
                    for i, session in enumerate(sessions[:10], 1):
                        print(f"\n{i}. {session['session_id'][:8]}...")
                        print(f"   Messages: {session['message_count']}")
                        print(f"   Last: {session['last_message'][:19]}")
                        print(f"   Preview: {session['preview']}...")
                else:
                    print("\nNo past conversations")
                continue
            
            if user_input.lower().startswith('switch '):
                session_id = user_input.split(' ', 1)[1]
                sessions = chat.list_past_sessions()
                full_id = next((s['session_id'] for s in sessions if s['session_id'].startswith(session_id)), None)
                if full_id:
                    chat.switch_session(full_id)
                else:
                    print(f"Not found: {session_id}")
                continue
            
            # Process
            response = chat.ask(user_input)
            print(f"\nBabyJay: {response}")
            
        except KeyboardInterrupt:
            print("\nRock Chalk! 🏀")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()