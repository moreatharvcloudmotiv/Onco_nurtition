import os
import json
import pandas as pd
import streamlit as st
import asyncio
import nest_asyncio
import threading
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import qdrant_client

# Fix for asyncio event loop issues in Streamlit
def setup_asyncio():
    """Setup asyncio event loop for the current thread"""
    try:
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
    except:
        pass
    
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Store loop in thread for later reference
        threading.current_thread().loop = loop
    except Exception as e:
        # Fallback: create new loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except:
            pass

# Initialize asyncio setup
setup_asyncio()

# =========== ENV & PAGE ===========
load_dotenv()
st.set_page_config(
    page_title="OncoNutrition - Smart Recipe Generator",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍽️"
)

# Custom CSS for dark theme with better visibility
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    /* Main content area */
    .main .block-container {
        background-color: #000000;
        color: #ffffff;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .step-header {
        background: #1e1e1e;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        border-radius: 5px;
        color: #ffffff;
    }

    .recipe-card {
        background: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
        color: #ffffff;
    }

    .recipe-card h3 {
        color: #ffffff;
        margin-bottom: 1rem;
    }

    .nutrition-box {
        background: #2d4a2d;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        color: #ffffff;
    }

    .meal-tab {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
    }

    .warning-box {
        background: #3d3d1f;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffc107;
    }

    /* All text elements */
    .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Markdown containers */
    div[data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    /* Form elements */
    .stTextInput > div > div > input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    .stSelectbox > div > div > div {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    .stTextArea > div > div > textarea {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    .stNumberInput > div > div > input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    .stMultiSelect > div > div > div {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    /* Labels */
    .stTextInput label, .stSelectbox label, .stTextArea label, .stNumberInput label, .stMultiSelect label {
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #667eea;
        color: #ffffff;
        border: none;
    }

    .stButton > button:hover {
        background-color: #5a6fd8;
    }

    /* Forms */
    .stForm {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #1e1e1e;
    }

    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Progress elements */
    .stProgress > div > div > div {
        background-color: #667eea;
    }

    /* Info boxes */
    .stInfo {
        background-color: #1e3a5f;
        color: #ffffff;
    }

    .stSuccess {
        background-color: #1e5f3a;
        color: #ffffff;
    }

    .stWarning {
        background-color: #5f4f1e;
        color: #ffffff;
    }

    .stError {
        background-color: #5f1e1e;
        color: #ffffff;
    }

    /* Metrics */
    .metric-container {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
    }

    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
    }

    /* Columns */
    .stColumn {
        background-color: #000000;
        color: #ffffff;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Code blocks */
    .stCode {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .stChatMessage div[data-testid="chatAvatarIcon-user"] {
        background-color: #667eea;
    }

    .stChatMessage div[data-testid="chatAvatarIcon-assistant"] {
        background-color: #4caf50;
    }

    /* Chat input */
    .stChatInput > div > div > input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
</style>""", unsafe_allow_html=True)

QDRANT_URL = os.getenv("QDRANT_URL") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
COLLECTION_NAME = "onco_recipes_google"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "models/gemini-1.5-flash"
CSV_PATH = "sample_dataset.csv"

# =========== SESSION STATE MANAGEMENT ===========
def init_session_state():
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'patient_profile' not in st.session_state:
        st.session_state.patient_profile = {}
    if 'generated_recipes' not in st.session_state:
        st.session_state.generated_recipes = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = None

# =========== HELPERS ===========
def create_qdrant_client():
    """Create Qdrant client with proper event loop handling"""
    try:
        # Ensure asyncio is properly set up
        setup_asyncio()
        
        # Create the Qdrant client
        client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return client
        
    except Exception as e:
        st.error(f"❌ Failed to create Qdrant client: {e}")
        # Try one more time with a fresh event loop
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            return client
        except Exception as e2:
            st.error(f"❌ Final attempt failed: {e2}")
            raise e2

def _safe_json_loads(s: str):
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
        return json.loads(s)
    except Exception:
        return None

def _render_recipe_card(recipe, meal_type):
    with st.container():
        recipe_name = recipe.get('name', 'Custom Recipe')
        st.markdown(f"""
<div class="recipe-card">
    <h3>&#127869; {recipe_name}</h3>
    <div class="meal-tab">{meal_type.upper()}</div>
</div>
        """, unsafe_allow_html=True)
        
        # Recipe metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            if recipe.get("servings"):
                st.metric("Servings", recipe['servings'])
        with col2:
            if recipe.get("prep_time"):
                st.metric("Prep Time", recipe['prep_time'])
        with col3:
            if recipe.get("total_time"):
                st.metric("Total Time", recipe['total_time'])
        
        # Goal alignment
        if recipe.get("goal_alignment"):
            st.info(f"**Why this works:** {recipe['goal_alignment']}")
        
        # Contraindications warning
        if recipe.get("contraindications") and recipe["contraindications"].lower() != "none known":
            contraindications = recipe['contraindications']
            st.markdown(f"""
<div class="warning-box">
    <strong>&#9888; Important Notes:</strong> {contraindications}
</div>
            """, unsafe_allow_html=True)
        
        # Two columns for ingredients and nutrition
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if recipe.get("ingredients"):
                st.subheader("🛒 Ingredients")
                for ing in recipe["ingredients"]:
                    st.write(f"• {ing}")
        
        with col2:
            if recipe.get("nutrition"):
                st.markdown('<div class="nutrition-box">', unsafe_allow_html=True)
                st.subheader("📊 Nutrition (per serving)")
                n = recipe["nutrition"]
                for key, value in n.items():
                    if key != "notes":
                        st.write(f"**{key.title()}:** {value}")
                if n.get("notes"):
                    st.caption(f"💡 {n['notes']}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Method
        if recipe.get("method_steps"):
            st.subheader("👨‍🍳 Method")
            for i, step in enumerate(recipe["method_steps"], 1):
                st.write(f"**{i}.** {step}")
        
        # Adaptations
        if recipe.get("adaptations"):
            with st.expander("🔄 Recipe Adaptations & Substitutions"):
                for adaptation in recipe["adaptations"]:
                    st.write(f"• {adaptation}")

# =========== RAG CORE ===========
class OncoNutritionRAG:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found in .env file.")
            raise ValueError("Missing GOOGLE_API_KEY")

        self.embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        
        # Use helper function to create Qdrant client with proper event loop handling
        self.client = create_qdrant_client()
        
        self.llm = GoogleGenerativeAI(model=LLM_MODEL, temperature=0.25, google_api_key=api_key)
        self.vs = self._get_or_create_vector_store()

    def _get_or_create_vector_store(self):
        try:
            info = self.client.get_collection(collection_name=COLLECTION_NAME)
            if getattr(info, "points_count", 0) > 0:
                st.success(f"✅ Connected to existing database with {info.points_count} recipes.")
                return Qdrant(self.client, COLLECTION_NAME, self.embedder)
        except Exception:
            st.info(f"📊 Initializing new recipe database...")

        if not os.path.exists(CSV_PATH):
            st.error(f"❌ Recipe database '{CSV_PATH}' not found.")
            return None

        with st.spinner("🔄 Building recipe database..."):
            df = pd.read_csv(CSV_PATH)
            docs = []
            for idx, row in df.iterrows():
                content = "\n".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip())
                
                # Enhanced metadata extraction for better retrieval
                metadata = {
                    "row_id": int(idx),
                    "cancer_type": str(row.get('CT', '')).lower(),
                    "cancer_stage": str(row.get('Cancer stage', '')).lower(),
                    "treatment_stage": str(row.get('Treatment stage', '')).lower(),
                    "allergies": str(row.get('Allergies', '')).lower(),
                    "adverse_effects": str(row.get('Adverse effects', '')).lower(),
                    "feeding_method": str(row.get('Methods', '')).lower(),
                    "gender": str(row.get('Gender', '')).lower(),
                    "age": str(row.get('Age', '')),
                }
                
                docs.append(Document(page_content=content, metadata=metadata))

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
            chunks = splitter.split_documents(docs)

            vs = Qdrant.from_documents(
                chunks,
                self.embedder,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME,
                force_recreate=True,
            )
            st.success(f"✅ Recipe database ready with {len(chunks)} entries.")
            return vs

    def retrieve_context(self, patient_profile: dict, meal_type: str, max_results: int = 8):
        """Enhanced retrieval that prioritizes cancer type and stage matches"""
        
        # Step 1: Try to find exact cancer type and stage matches first
        # Normalize cancer type to match CSV format (single word)
        cancer_type_map = {
            'breast cancer': 'breast',
            'lung cancer': 'lung',
            'colorectal cancer': 'colorectal',
            'prostate cancer': 'prostate'
        }
        cancer_type_normalized = cancer_type_map.get(patient_profile['cancer_type'].lower(),
                                                     patient_profile['cancer_type'].lower().replace(' cancer', '').replace(' ', ''))
        
        # Normalize stage to match CSV format
        stage_normalized = patient_profile['cancer_stage'].lower().replace(' (early)', '').replace(' (localized)', '').replace(' (regional)', '').replace(' (metastatic)', '').replace(' (in situ)', '')
        
        # Create hierarchical search queries
        searches = [
            # Most specific: exact cancer type + stage + meal type
            f"CT {cancer_type_normalized} {stage_normalized} {meal_type}",
            # Cancer type + stage (any meal)
            f"CT {cancer_type_normalized} {stage_normalized}",
            # Cancer type + treatment stage
            f"CT {cancer_type_normalized} {patient_profile['treatment_stage'].lower()}",
            # Cancer type only
            f"CT {cancer_type_normalized}",
            # Treatment stage + symptoms
            f"{patient_profile['treatment_stage'].lower()} {patient_profile['current_symptoms']}",
            # Feeding method + general
            f"{patient_profile['feeding_method'].lower()} {meal_type}",
        ]
        
        # Collect results with scoring
        all_docs = []
        seen_row_ids = set()
        
        for priority, query in enumerate(searches):
            if len(all_docs) >= max_results:
                break
                
            try:
                retriever = self.vs.as_retriever(search_kwargs={"k": min(max_results*2, 20)})
                docs = retriever.get_relevant_documents(query)
                
                for doc in docs:
                    row_id = doc.metadata.get('row_id')
                    if row_id not in seen_row_ids and len(all_docs) < max_results:
                        # Add priority score to metadata
                        doc.metadata['priority_score'] = len(searches) - priority
                        doc.metadata['search_query'] = query
                        all_docs.append(doc)
                        seen_row_ids.add(row_id)
                        
            except Exception as e:
                st.warning(f"Search query failed: {query}")
                continue
        
        # If we still don't have enough, do a broader search
        if len(all_docs) < max_results:
            fallback_query = f"{patient_profile['cancer_type']} {meal_type} nutrition"
            try:
                retriever = self.vs.as_retriever(search_kwargs={"k": max_results})
                fallback_docs = retriever.get_relevant_documents(fallback_query)
                
                for doc in fallback_docs:
                    row_id = doc.metadata.get('row_id')
                    if row_id not in seen_row_ids and len(all_docs) < max_results:
                        doc.metadata['priority_score'] = 0
                        doc.metadata['search_query'] = fallback_query
                        all_docs.append(doc)
                        seen_row_ids.add(row_id)
            except Exception:
                pass
        
        # Sort by priority score (higher = better match)
        all_docs.sort(key=lambda x: x.metadata.get('priority_score', 0), reverse=True)
        
        return all_docs[:max_results]

    def generate_custom_recipes(self, patient_profile: dict, meal_type: str, retrieved_docs: list, n_recipes: int = 3):
        # Organize sources by priority for better prompting
        exact_matches = [d for d in retrieved_docs if d.metadata.get('priority_score', 0) >= 5]
        good_matches = [d for d in retrieved_docs if 3 <= d.metadata.get('priority_score', 0) < 5]
        related_sources = [d for d in retrieved_docs if d.metadata.get('priority_score', 0) < 3]
        
        snippets = []
        
        # Prioritize exact matches
        if exact_matches:
            snippets.append("=== EXACT CANCER TYPE/STAGE MATCHES (Priority Sources) ===")
            for d in exact_matches:
                text = d.page_content.strip()
                if len(text) > 1200:
                    text = text[:1200] + " …"
                metadata = d.metadata
                snippets.append(f"[EXACT MATCH - row_id={metadata.get('row_id','?')} - {metadata.get('cancer_type','')} {metadata.get('cancer_stage','')}]\n{text}")
        
        if good_matches:
            snippets.append("\n=== GOOD MATCHES ===")
            for d in good_matches:
                text = d.page_content.strip()
                if len(text) > 1000:
                    text = text[:1000] + " …"
                metadata = d.metadata
                snippets.append(f"[GOOD MATCH - row_id={metadata.get('row_id','?')} - {metadata.get('cancer_type','')}]\n{text}")
        
        if related_sources and len(snippets) < 3:  # Only use related if we don't have enough good matches
            snippets.append("\n=== RELATED SOURCES ===")
            for d in related_sources[:2]:  # Limit related sources
                text = d.page_content.strip()
                if len(text) > 800:
                    text = text[:800] + " …"
                snippets.append(f"[RELATED - row_id={d.metadata.get('row_id','?')}]\n{text}")

        meal_guidance = {
            "breakfast": f"Focus on gentle, easy-to-digest options for {patient_profile['feeding_method']}. Consider nutritional supplements if appetite is poor in the morning. Adapt for {patient_profile['physical_health']} physical condition.",
            "lunch": f"Balanced meals with good protein sources suitable for {patient_profile['feeding_method']}. Consider {patient_profile['mental_health']} mental state and energy levels.",
            "dinner": f"Lighter options if treatment affects evening appetite, adapted for {patient_profile['feeding_method']}. Consider timing with medication schedules.",
            "snack": f"Small, nutrient-dense options for {patient_profile['feeding_method']} that can help maintain energy between meals or during treatment."
        }

        system_prompt = f"""
You are a clinical nutrition assistant specializing in oncology. Create {meal_type.upper()} recipes specifically.

PATIENT CONTEXT:
- Feeding method: {patient_profile['feeding_method']}
- Cancer type: {patient_profile['cancer_type']} ({patient_profile['cancer_stage']})
- Treatments: {', '.join(patient_profile.get('treatment_types', []))}
- Physical health: {patient_profile['physical_health']}
- Mental health: {patient_profile['mental_health']}
- Current symptoms: {patient_profile['current_symptoms']}

MEAL TYPE GUIDANCE: {meal_guidance.get(meal_type, "")}

CRITICAL ADAPTATIONS:
- ALL recipes MUST be suitable for {patient_profile['feeding_method']}
- Consider dietary restrictions: {', '.join(patient_profile.get('dietary_restrictions', []))}
- Avoid allergens: {patient_profile.get('allergies_str', 'none')}

PRIORITIZE EXACT MATCHES: Use the "EXACT CANCER TYPE/STAGE MATCHES" sources first as they are most relevant to this patient's condition.

Base your recipes ONLY on the provided context. Create NEW, CUSTOMIZED {meal_type} recipes that are safe and practical.

Return STRICT JSON in this schema:
{{
  "patient_summary": "1-3 sentence summary for this {meal_type} considering feeding method",
  "recipes": [
    {{
      "name": "string (include meal type in name)",
      "servings": "string or number",
      "prep_time": "e.g., 10 min",
      "cook_time": "e.g., 15 min",
      "total_time": "e.g., 25 min",
      "ingredients": ["qty + ingredient + form (adapted for {patient_profile['feeding_method']})"],
      "method_steps": ["Step 1 (adapted for feeding method)", "Step 2"],
      "nutrition": {{
        "calories": "approx per serving",
        "protein": "g",
        "carbs": "g",
        "fat": "g",
        "fiber": "g",
        "notes": "micronutrients note"
      }},
      "goal_alignment": "how this {meal_type} supports the patient's specific condition and feeding method",
      "adaptations": ["modifications for {patient_profile['feeding_method']}", "texture modifications", "dietary restrictions handling"],
      "contraindications": "what to avoid considering feeding method or 'none known'"
    }}
  ]
}}
"""

        patient_block = json.dumps(patient_profile, ensure_ascii=False, indent=2)
        context_block = "\n\n".join(snippets) if snippets else "NO CONTEXT"

        user_prompt = f"""
PATIENT PROFILE:
{patient_block}

RETRIEVED CONTEXT:
{context_block}
"""

        response = self.llm.invoke(system_prompt + "\n\n" + user_prompt)
        return _safe_json_loads(response)

    def chat_with_context(self, patient_profile: dict, user_message: str, chat_history: list = None):
        """Chat interface with nutritional context"""
        
        # Get relevant context based on user message and patient profile
        docs = self.retrieve_context(patient_profile, user_message, max_results=5)
        
        # Prepare context snippets
        context_snippets = []
        for d in docs[:3]:  # Limit to top 3 for chat
            text = d.page_content.strip()
            if len(text) > 800:
                text = text[:800] + "..."
            context_snippets.append(f"[Source: {d.metadata.get('cancer_type', 'unknown')} {d.metadata.get('cancer_stage', '')}]\n{text}")
        
        # Build chat history context
        history_context = ""
        if chat_history:
            recent_history = chat_history[-6:]  # Last 3 exchanges
            for msg in recent_history:
                role = "Human" if msg["role"] == "user" else "Assistant"
                history_context += f"{role}: {msg['content']}\n"
        
        system_prompt = f"""You are a helpful and supportive oncology nutrition assistant providing personalized advice through a chat interface. Your tone is warm and empathetic.
        
        PATIENT CONTEXT:
        - Name: {patient_profile.get('name', 'Patient')}
        - Cancer: {patient_profile.get('cancer_type', 'Unknown')} ({patient_profile.get('cancer_stage', 'Unknown stage')})
        - Treatment stage: {patient_profile.get('treatment_stage', 'Unknown')}
        - Feeding method: {patient_profile.get('feeding_method', 'Unknown')}
        - Current symptoms: {patient_profile.get('current_symptoms', 'None specified')}
        - Allergies: {patient_profile.get('allergies_str', 'None')}
        - Dietary restrictions: {', '.join(patient_profile.get('dietary_restrictions', []))}

        CRITICAL RESPONSE FORMAT:
        - Your response MUST be exactly ONE paragraph only - no exceptions
        - Maximum 3-4 sentences per response
        - Do NOT use bullet points, lists, or multiple paragraphs
        - Be direct, practical, and supportive in a single flowing paragraph
        - Include specific advice tailored to their cancer type, stage, and feeding method
        - If medical advice is needed, mention consulting their healthcare team within the same paragraph

        IMPORTANT: Always adapt advice for their feeding method: {patient_profile.get('feeding_method', 'oral')}"""

        user_prompt = f"""
    RECENT CONVERSATION:
    {history_context}

    RELEVANT NUTRITIONAL CONTEXT:
    {chr(10).join(context_snippets) if context_snippets else "No specific context found"}

    CURRENT QUESTION: {user_message}

    Please provide a helpful, personalized response considering their cancer type, treatment stage, and feeding method."""

        response = self.llm.invoke(system_prompt + "\n\n" + user_prompt)
        return response

# =========== UI SCREENS ===========
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ OncoNutrition</h1>
        <p>Personalized Recipe Generator for Cancer Care</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_navigation():
    st.sidebar.title("🏥 Navigation")
    
    # Progress indicator
    steps = ["Patient Info", "Meal Selection", "Recipe Generation", "Results", "Chat Assistant"]
    for i, step in enumerate(steps, 1):
        if i == st.session_state.current_step:
            st.sidebar.markdown(f"**▶️ {i}. {step}**")
        elif i < st.session_state.current_step:
            st.sidebar.markdown(f"✅ {i}. {step}")
        else:
            st.sidebar.markdown(f"⏳ {i}. {step}")
    
    st.sidebar.divider()
    
    # Navigation buttons
    if st.session_state.current_step > 1:
        if st.sidebar.button("⬅️ Previous Step"):
            st.session_state.current_step -= 1
            st.rerun()
    
    if st.session_state.current_step < 5:
        if st.sidebar.button("➡️ Next Step"):
            st.session_state.current_step += 1
            st.rerun()
    
    # Quick navigation to chat
    if st.session_state.patient_profile and st.sidebar.button("💬 Jump to Chat"):
        st.session_state.current_step = 5
        st.rerun()
    
    if st.sidebar.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Debug: Clear cache if needed
    if st.sidebar.button("🗑️ Clear Cache", help="Clear cached RAG system"):
        clear_rag_cache()
        st.sidebar.success("Cache cleared!")

def screen_patient_info():
    st.markdown('<div class="step-header"><h2>👤 Step 1: Patient Information</h2></div>', unsafe_allow_html=True)
    
    with st.form("patient_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Basic Information")
            name = st.text_input("Patient Name", value=st.session_state.patient_profile.get('name', ''))
            age = st.number_input("Age", 0, 120, st.session_state.patient_profile.get('age', 60))
            gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                 index=["Male", "Female", "Other"].index(st.session_state.patient_profile.get('gender', 'Male')))
            
            st.subheader("🩺 Cancer Information")
            cancer_type = st.selectbox("Cancer Type",
                                      ["Breast Cancer", "Lung Cancer", "Colorectal Cancer", "Prostate Cancer"],
                                      index=0 if 'cancer_type' not in st.session_state.patient_profile else
                                      ["Breast Cancer", "Lung Cancer", "Colorectal Cancer", "Prostate Cancer"].index(
                                          st.session_state.patient_profile.get('cancer_type', 'Breast Cancer')))
            
            cancer_stage = st.selectbox("Cancer Stage",
                                       ["Stage 0 (In Situ)", "Stage I (Early)", "Stage II (Localized)",
                                        "Stage III (Regional)", "Stage IV (Metastatic)", "Unknown"],
                                       index=st.session_state.patient_profile.get('cancer_stage_index', 0))
        
        with col2:
            st.subheader("🏥 Treatment Information")
            treatment_types = st.multiselect(
                "Current/Recent Treatments (Select all that apply)",
                ["Chemotherapy", "Radiation Therapy", "Surgery", "Immunotherapy",
                 "Hormone Therapy", "Targeted Therapy", "Stem Cell Transplant", "Other"],
                default=st.session_state.patient_profile.get('treatment_types', [])
            )
            
            treatment_stage = st.selectbox(
                "Treatment Stage",
                ["Pre-treatment", "Active Treatment", "Between Treatments",
                 "Post-treatment Recovery", "Long-term Survivorship", "Palliative Care"],
                index=["Pre-treatment", "Active Treatment", "Between Treatments",
                        "Post-treatment Recovery", "Long-term Survivorship", "Palliative Care"].index(
                    st.session_state.patient_profile.get('treatment_stage', 'Pre-treatment'))
            )
            
            st.subheader("💪 Patient Health Status")
            physical_health = st.selectbox("Physical Health Status",
                                         ["Excellent", "Good", "Fair", "Poor", "Critical"],
                                         index=["Excellent", "Good", "Fair", "Poor", "Critical"].index(
                                             st.session_state.patient_profile.get('physical_health', 'Good')))
            
            mental_health = st.selectbox("Mental Health Status",
                                       ["Excellent", "Good", "Fair", "Poor", "Seeking Support"],
                                       index=["Excellent", "Good", "Fair", "Poor", "Seeking Support"].index(
                                           st.session_state.patient_profile.get('mental_health', 'Good')))
            
            current_symptoms = st.text_area("Adverse Effects",
                                         value=st.session_state.patient_profile.get('current_symptoms', ''),
                                         help="e.g., nausea, fatigue, loss of appetite, taste changes, mouth sores")
        
        # Second row for feeding method and dietary restrictions
        st.subheader("🍽️ Feeding and Dietary Information")
        col3, col4 = st.columns(2)
        
        with col3:
            feeding_method = st.selectbox("Method of Food Intake",
                                         ["Oral (Normal eating)", "Soft/Modified diet", "Liquid diet only",
                                          "Feeding Tube (Nasogastric)", "Feeding Tube (PEG/Gastrostomy)",
                                          "Feeding Tube (Jejunostomy)", "IV Nutrition (TPN)", "Combined methods"],
                                         index=["Oral (Normal eating)", "Soft/Modified diet", "Liquid diet only",
                                                 "Feeding Tube (Nasogastric)", "Feeding Tube (PEG/Gastrostomy)",
                                                 "Feeding Tube (Jejunostomy)", "IV Nutrition (TPN)", "Combined methods"].index(
                                                st.session_state.patient_profile.get('feeding_method', 'Oral (Normal eating)')))
            
            allergies = st.text_input("Food Allergies",
                                     value=st.session_state.patient_profile.get('allergies_str', ''),
                                     help="Separate multiple allergies with commas")
        
        with col4:
            dietary_restrictions = st.multiselect("Dietary Restrictions/Preferences",
                                                 ["Vegetarian", "Vegan", "Gluten-free", "Dairy-free",
                                                  "Low-sodium", "Diabetic diet", "Heart-healthy",
                                                  "Kosher", "Halal", "No restrictions"],
                                                 default=st.session_state.patient_profile.get('dietary_restrictions', []))
            
            cultural_preferences = st.text_input("Cultural/Cuisine Preferences",
                                                 value=st.session_state.patient_profile.get('cultural_preferences_str', ''),
                                                 help="e.g., Mediterranean, Asian, Mexican, comfort foods")
        
        if st.form_submit_button("💾 Save Patient Information", type="primary"):
            st.session_state.patient_profile = {
                'name': name,
                'age': age,
                'gender': gender,
                'cancer_type': cancer_type,
                'cancer_stage': cancer_stage,
                'cancer_stage_index': ["Stage 0 (In Situ)", "Stage I (Early)", "Stage II (Localized)",
                                       "Stage III (Regional)", "Stage IV (Metastatic)", "Unknown"].index(cancer_stage),
                'treatment_types': treatment_types,
                'treatment_stage': treatment_stage,
                'physical_health': physical_health,
                'mental_health': mental_health,
                'current_symptoms': current_symptoms,
                'feeding_method': feeding_method,
                'allergies': [a.strip() for a in allergies.split(",") if a.strip()],
                'allergies_str': allergies,
                'dietary_restrictions': dietary_restrictions,
                'cultural_preferences': [c.strip() for c in cultural_preferences.split(",") if c.strip()],
                'cultural_preferences_str': cultural_preferences
            }
            st.success("✅ Patient information saved!")
            st.session_state.current_step = 2
            st.rerun()

def screen_meal_selection():
    st.markdown('<div class="step-header"><h2>🍽️ Step 2: Meal Planning</h2></div>', unsafe_allow_html=True)
    
    if not st.session_state.patient_profile:
        st.warning("⚠️ Please complete patient information first.")
        if st.button("⬅️ Go to Patient Info"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    # Display patient summary
    profile = st.session_state.patient_profile
    st.info(f"👤 Planning meals for {profile['name']} ({profile['age']} years old) with {profile['cancer_type']} - {profile['feeding_method']}")
    
    with st.form("meal_selection_form"):
        st.subheader("Select Meal Types")
        
        meal_cols = st.columns(4)
        meal_types = []
        
        with meal_cols[0]:
            if st.checkbox("🌅 Breakfast", help="Light, energizing morning meals"):
                meal_types.append("breakfast")
        
        with meal_cols[1]:
            if st.checkbox("🌞 Lunch", help="Balanced midday meals"):
                meal_types.append("lunch")
        
        with meal_cols[2]:
            if st.checkbox("🌙 Dinner", help="Lighter evening meals"):
                meal_types.append("dinner")
        
        with meal_cols[3]:
            if st.checkbox("🍎 Snacks", help="Small, nutrient-dense options"):
                meal_types.append("snack")
        
        st.subheader("Recipe Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            n_recipes = st.slider("Recipes per meal type", 1, 4, 2)
        
        with col2:
            complexity = st.selectbox("Recipe Complexity",
                                     ["Simple (5-10 ingredients)", "Moderate (10-15 ingredients)", "Complex (15+ ingredients)"])
        
        advanced_options = st.expander("⚙️ Advanced Options")
        with advanced_options:
            top_k = st.slider("Database search depth", 3, 15, 8,
                             help="Higher values search more recipes for better matches")
            include_context = st.checkbox("Show recipe sources",
                                         help="Display the database entries used to create recipes")
        
        if st.form_submit_button("🚀 Generate Meal Plan", type="primary"):
            if not meal_types:
                st.error("❌ Please select at least one meal type.")
                return
            
            st.session_state.meal_config = {
                'meal_types': meal_types,
                'n_recipes': n_recipes,
                'complexity': complexity,
                'top_k': top_k,
                'include_context': include_context
            }
            st.session_state.current_step = 3
            st.rerun()

def screen_recipe_generation():
    st.markdown('<div class="step-header"><h2>🧑‍🍳 Step 3: Generating Your Recipes</h2></div>', unsafe_allow_html=True)
    
    if 'meal_config' not in st.session_state:
        st.warning("⚠️ Please select meal preferences first.")
        if st.button("⬅️ Go to Meal Selection"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    # Add cache clear option for debugging
    if st.button("🗑️ Clear Cache and Reinitialize", help="Click if you encounter errors"):
        clear_rag_cache()
        st.success("Cache cleared! Please try again.")
        st.stop()
    
    # Initialize RAG system
    rag = get_rag()
    if not rag or not rag.vs:
        st.error("❌ Recipe database not available. Please check your configuration.")
        return
    
    config = st.session_state.meal_config
    profile = st.session_state.patient_profile
    
    # Generate recipes for each meal type
    all_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, meal_type in enumerate(config['meal_types']):
        status_text.text(f"🔍 Finding {meal_type} recipes...")
        progress_bar.progress((i + 0.5) / len(config['meal_types']))
        
        # Retrieve context using enhanced matching
        docs = rag.retrieve_context(profile, meal_type, config['top_k'])
        
        status_text.text(f"🧑‍🍳 Creating custom {meal_type} recipes...")
        
        # Generate recipes
        result = rag.generate_custom_recipes(
            profile, meal_type, docs, n_recipes=config['n_recipes']
        )
        
        if result:
            all_results[meal_type] = {
                'result': result,
                'context': docs if config['include_context'] else []
            }
        
        progress_bar.progress((i + 1) / len(config['meal_types']))
    
    status_text.text("✅ Recipe generation complete!")
    
    st.session_state.generated_recipes = all_results
    st.session_state.current_step = 4
    
    # Auto-advance to results
    st.success("🎉 Your personalized meal plan is ready!")
    if st.button("📋 View Results", type="primary"):
        st.rerun()

def screen_results():
    st.markdown('<div class="step-header"><h2>📋 Step 4: Your Personalized Meal Plan</h2></div>', unsafe_allow_html=True)
    
    if not st.session_state.generated_recipes:
        st.warning("⚠️ No recipes generated yet.")
        if st.button("⬅️ Go to Generation"):
            st.session_state.current_step = 3
            st.rerun()
        return
    
    # Display patient summary
    profile = st.session_state.patient_profile
    st.success(f"🎯 Meal plan for {profile['name']} - {profile['cancer_type']} ({profile['cancer_stage']}) - {profile['feeding_method']}")
    
    # Meal type tabs
    meal_tabs = st.tabs([f"{meal.title()}s" for meal in st.session_state.generated_recipes.keys()])
    
    for tab, (meal_type, data) in zip(meal_tabs, st.session_state.generated_recipes.items()):
        with tab:
            result = data['result']
            
            # Show matching statistics
            if data['context']:
                exact_matches = sum(1 for doc in data['context'] if doc.metadata.get('priority_score', 0) >= 5)
                good_matches = sum(1 for doc in data['context'] if 3 <= doc.metadata.get('priority_score', 0) < 5)
                total_sources = len(data['context'])
                
                if exact_matches > 0:
                    st.success(f"🎯 Found {exact_matches} exact cancer type/stage matches out of {total_sources} sources")
                elif good_matches > 0:
                    st.info(f"✅ Found {good_matches} good matches out of {total_sources} sources")
                else:
                    st.warning(f"🔍 Using {total_sources} related sources (no exact cancer type/stage matches found)")
            
            # Patient summary for this meal
            if result.get('patient_summary'):
                st.info(f"💡 **{meal_type.title()} Focus:** {result['patient_summary']}")
            
            # Display recipes
            recipes = result.get('recipes', [])
            if recipes:
                for i, recipe in enumerate(recipes):
                    _render_recipe_card(recipe, meal_type)
                    if i < len(recipes) - 1:
                        st.divider()
            else:
                st.warning(f"No {meal_type} recipes were generated.")
            
            # Show context if requested
            if data['context'] and st.session_state.meal_config.get('include_context'):
                with st.expander(f"🔍 Recipe Sources for {meal_type.title()} (Ordered by Relevance)"):
                    for i, doc in enumerate(data['context']):
                        priority = doc.metadata.get('priority_score', 0)
                        search_query = doc.metadata.get('search_query', 'Unknown')
                        cancer_type = doc.metadata.get('cancer_type', 'unknown')
                        cancer_stage = doc.metadata.get('cancer_stage', 'unknown')
                        
                        # Show match quality
                        match_quality = "🎯 Exact Match" if priority >= 5 else "✅ Good Match" if priority >= 3 else "🔍 Related"
                        
                        st.markdown(f"""
                        **Source #{i+1}** {match_quality} - Priority: {priority}
                        - **Cancer Type:** {cancer_type.title()}
                        - **Stage:** {cancer_stage.title()}
                        - **Found via:** {search_query}
                        - **Row ID:** `{doc.metadata.get('row_id', 'N/A')}`
                        """)
                        st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        if i < len(data['context']) - 1:
                            st.divider()
    
    # Action buttons
    st.divider()
    col4 = st.columns(1)[0]
    
    with col4:
        if st.button("💬 Chat Assistant"):
            st.session_state.current_step = 5
            st.rerun()

def screen_chat():
    st.markdown('<div class="step-header"><h2>💬 Step 5: Nutrition Chat Assistant</h2></div>', unsafe_allow_html=True)
    
    if not st.session_state.patient_profile:
        st.warning("⚠️ Please complete patient information first.")
        if st.button("⬅️ Go to Patient Info"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    # Initialize RAG system
    rag = get_rag()
    if not rag or not rag.vs:
        st.error("❌ Chat system not available. Please check your configuration.")
        return
    
    profile = st.session_state.patient_profile
    
    # Display patient context
    st.info(f"💬 Chatting as {profile['name']} - {profile['cancer_type']} ({profile['cancer_stage']}) - {profile['feeding_method']}")
    
    # Chat interface
    st.subheader("Ask me anything about nutrition, recipes, or dietary concerns!")
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Quick question buttons
    st.markdown("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = {
        "🤢 Managing Nausea": "I'm experiencing nausea from my treatment. What foods should I eat or avoid?",
        "😴 Fatigue & Energy": "I feel very tired all the time. What foods can help boost my energy?",
        "🍽️ Appetite Loss": "I've lost my appetite. How can I make sure I'm getting enough nutrition?",
        "👅 Taste Changes": "Food tastes different or metallic. Any suggestions for making meals more appealing?"
    }

    # Handle quick question buttons
    for i, (label, question) in enumerate(quick_questions.items()):
        with col1 if i == 0 else col2 if i == 1 else col3 if i == 2 else col4:
            if st.button(label):
                st.session_state.chat_messages.append({"role": "user", "content": question})
                # Set a flag to process the quick question
                st.session_state.process_quick_question = True
                st.rerun()

    # Chat input and message processing
    if prompt := st.chat_input("Type your nutrition question here..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.session_state.process_quick_question = True
        st.rerun()

    # Process new messages (either from chat input or quick buttons)
    if st.session_state.get('process_quick_question', False):
        user_message = st.session_state.chat_messages[-1]['content']
        with st.chat_message("user"):
            st.markdown(user_message)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag.chat_with_context(
                    patient_profile=profile,
                    user_message=user_message,
                    chat_history=st.session_state.chat_messages[:-1]
                )
            st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Reset the flag after processing
        st.session_state.process_quick_question = False
        st.rerun()

    # Chat controls
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("⬅️ Back to Results"):
            st.session_state.current_step = 4
            st.rerun()
    
    with col3:
        if st.button("📋 Generate More Recipes"):
            st.session_state.current_step = 2
            st.rerun()

# =========== MAIN APP ===========
@st.cache_resource
def get_rag(version="v2.0"):  # Added version to force cache refresh
    try:
        return OncoNutritionRAG()
    except Exception as e:
        st.error(f"❌ Initialization error: {e}")
        return None

def clear_rag_cache():
    """Clear the cached RAG instance"""
    get_rag.clear()

def main():
    init_session_state()
    render_header()
    render_sidebar_navigation()
    
    # Route to appropriate screen
    if st.session_state.current_step == 1:
        screen_patient_info()
    elif st.session_state.current_step == 2:
        screen_meal_selection()
    elif st.session_state.current_step == 3:
        screen_recipe_generation()
    elif st.session_state.current_step == 4:
        screen_results()
    elif st.session_state.current_step == 5:
        screen_chat()

if __name__ == "__main__":
    main()
