import streamlit as st
import os, docx2txt, asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from crewai import Agent, Crew, Process, Task
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool

# 1. SETUP ENVIRONMENTS & LLM ACCELERATION HOOKS
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# Safeguard Serper API Key into the environment dictionary where LangChain wrapper expects it
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY

gemini_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3) if GOOGLE_API_KEY else None
groq_llm = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=GROQ_API_KEY, temperature=0.4, max_tokens=800) if GROQ_API_KEY else None

if GOOGLE_API_KEY:
    st.sidebar.success("🤖 Primary Engine: Gemini Active")
elif GROQ_API_KEY:
    st.sidebar.warning("⚡ Backup Engine: Groq Qwen Active")
if not SERPER_API_KEY:
    st.sidebar.error("⚠️ SERPER_API_KEY environment variable missing!")

# FIXED: Utilizing official wrapper class to guarantee correct endpoint routing
@tool("Google Live Job Search Tool")
def search_tool(query: str) -> str:
    """Searches Google for real-time live job postings those are direct match or very close for the resume title, active hiring boards, and vacancy text snippets."""
    if not os.environ.get("SERPER_API_KEY", ""):
        return "Error: SERPER_API_KEY environment variable is missing."
    
    user_region = st.session_state.get('target_region', 'Montreal')
    clean_query = query.replace('"', '').replace("'", "")
    
    try:
        # Instantiating the official wrapper automatically handles endpoint routing
        search = GoogleSerperAPIWrapper()
        optimized_query = f"{clean_query} jobs vacancies related to the resume title hiring openings in {user_region} site:://linkedin.com OR site:indeed.com"
        return search.run(optimized_query)
    except Exception as e:
        return f"Serper API execution fault encountered: {str(e)}"

# 2. STREAMLIT WEB INTERFACE LAYOUT
st.set_page_config(page_title="AI Agent Career Strategist", layout="wide")
st.title("💼 AI Agent Career Strategist Network")
st.subheader("Optimize your resume using real-time market data")

col1, col2 = st.columns(2)
with col1:
    region = st.text_input("📍 Target Geographic Region", placeholder="e.g., Montreal", key="target_region", value="Montreal")
with col2:
    timeframe = st.selectbox("⏳ Job Posting Timeframe", ["Past 24 hours", "Past week", "Past month"])

resume_method = st.radio("Provide resume via:", ["Paste Text", "Upload Word Document (.docx)"])
resume_content = ""

if resume_method == "Paste Text":
    resume_content = st.text_area("Ctrl+V your resume here:", height=250)
else:
    uploaded_file = st.file_uploader("Upload your resume Word file", type=["docx"])
    if uploaded_file is not None:
        try:
            resume_content = docx2txt.process(uploaded_file)
            st.success("✅ Word document parsed successfully!")
        except Exception as e:
            st.error(f"Error parsing Word file: {e}")

# 3. HIGH-QUALITY AGENT ENGINE CORES
def create_specialized_agents(selected_llm):
    researcher = Agent(
        role='Market Research Analyst',
        goal=f'Analyze real-time search results of open job postings in {region} from the {timeframe}. Log exact vacancy titles and cross-reference common instances.',
        backstory="A precise labor market trends analyst who reads live Google indexing snippets to discover genuine positions.",
        verbose=True, allow_delegation=False, tools=[search_tool], llm=selected_llm
    )
    gap_analyst = Agent(
        role='Resume Gap Analyst',
        goal='Compare a technical resume against active market demands to uncover critical missing keywords and skills.',
        backstory="An expert ATS filtering specialist who exposes missing core assets in professional profiles.",
        verbose=True, allow_delegation=False, llm=selected_llm
    )
    education_advisor = Agent(
        role='Executive Education Consultant',
        goal='Design actionable study roadmaps, including real certifications and free learning materials.',
        backstory="A professional corporate learning curator who maps academic and skill development milestones.",
        verbose=True, allow_delegation=False, llm=selected_llm
    )
    return researcher, gap_analyst, education_advisor

# 4. STREAMLIT TRIGGER BLOCK (WITH DYNAMIC SECTION RENDERING)
if st.button("🚀 Analyze Market & Enhance Resume"):
    if not resume_content or not region:
        st.error("Please provide both your resume and a target region.")
    else:
        status_slot = st.empty()
        p1_slot = st.empty()
        p2_slot = st.empty()
        p3_slot = st.empty()
        
        max_words = 2000
        words = resume_content.split()
        safe_resume = " ".join(words[:max_words]) if len(words) > max_words else resume_content

        with st.spinner("Processing analysis pipeline..."):
            try:
                if gemini_llm:
                    status_slot.info("🔄 Querying Gemini 3.8 Flash...")
                    researcher, gap_analyst, _ = create_specialized_agents(gemini_llm)
                    
                    t_r = Task(description=f"Search Google via tool for active openings in {region}. Find the 5 most common position names and top 5 required skills.", expected_output="Live market data report with counts and job titles.", agent=researcher)
                    t_s = Task(description=f"Review resume:\n---\n{safe_resume}\n---\nCompare it against research data to format a full gap report and study path.", expected_output="Full strategy report.", agent=gap_analyst)
                    
                    crew = Crew(agents=[researcher, gap_analyst], tasks=[t_r, t_s], process=Process.sequential)
                    monolithic_result = str(crew.kickoff())
                    
                    status_slot.success("✨ Analysis Complete!")
                    p1_slot.markdown("### 📊 Market & Resume Strategy Report")
                    p1_slot.markdown(monolithic_result)
                else:
                    raise Exception("429 Quota triggered.")
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or not gemini_llm:
                    status_slot.warning("⚠️ Primary limit reached. Running incremental fallback sequence to prevent cut reports...")
                    
                    if groq_llm:
                        researcher, gap_analyst, education_advisor = create_specialized_agents(groq_llm)
                        
                        # PHASE 1
                        status_slot.info("⚡ Phase 1/3: Parsing Live Job Volumes...")
                        task_1 = Task(description=f"Analyze live search snippets for {region}. Review titles, count total openings found, and compile the 5 most frequent exact position names.", expected_output="Markdown table of counts and top 5 job titles.", agent=researcher)
                        phase_1_output = str(Crew(agents=[researcher], tasks=[task_1], process=Process.sequential).kickoff())
                        p1_slot.markdown(f"### 📊 1. Position Market Overview\n{phase_1_output}")
                        
                        # PHASE 2
                        status_slot.info("⚡ Phase 2/3: Uncovering Keyword Deficiencies...")
                        task_2 = Task(description=f"Compare this resume:\n---\n{safe_resume}\n---\nAgainst these openings:\n{phase_1_output}\nIdentify exactly which critical skills are missing.", expected_output="A structured list of core skill gaps.", agent=gap_analyst)
                        phase_2_output = str(Crew(agents=[gap_analyst], tasks=[task_2], process=Process.sequential).kickoff())
                        p2_slot.markdown(f"### 🔍 2. Resume Gap Analysis\n{phase_2_output}")
                        
                        # PHASE 3
                        status_slot.info("⚡ Phase 3/3: Drafting Study Roadmaps...")
                        task_3 = Task(description=f"Based on these gaps:\n{phase_2_output}\nProvide certification names and study materials.", expected_output="An actionable learning path with learning resources.", agent=education_advisor)
                        phase_3_output = str(Crew(agents=[education_advisor], tasks=[task_3], process=Process.sequential).kickoff())
                        p3_slot.markdown(f"### 🎓 3. Tailored Learning Path\n{phase_3_output}")
                        
                        status_slot.success("✨ All phases complete! Information saved smoothly.")
                    else:
                        status_slot.error("❌ Missing backup API configurations.")
                else:
                    status_slot.error(f"Unexpected error: {error_msg}")
