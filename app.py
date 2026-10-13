import streamlit as st
import os
import docx2txt
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from crewai import Agent, Crew, Process, Task
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# 1. SETUP FREE LLM CONFIGURATION WITH DYNAMIC FALLBACKS
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Primary engine
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
) if GOOGLE_API_KEY else None

# Fallback engine 
groq_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.4,
    max_tokens=800
) if GROQ_API_KEY else None

# Sidebar status
if GOOGLE_API_KEY:
    st.sidebar.success("🤖 Primary Engine: Gemini 3.8 Flash Active")
elif GROQ_API_KEY:
    st.sidebar.warning("⚡ Backup Engine: Groq Qwen 3.8 Active")

# FIXED: Re-engineered search tool that dynamically restructures queries to pull real vacancy pages
@tool("Web Search Tool")
def search_tool(query: str) -> str:
    """Useful to search the internet for live job postings, market trends, and role vacancies."""
    search = DuckDuckGoSearchRun()
    
    # Extract the user's targeted region from the Streamlit state securely
    user_region = st.session_state.get('target_region', 'Remote')
    
    # Force search operators to bypass generic blog posts and hit real job vacancy indicators
    optimized_query = f"{query} {user_region} site:://linkedin.com OR site:indeed.com jobs"
    
    try:
        return search.run(optimized_query)
    except Exception:
        # Secondary backup clean query if operators trigger strict DuckDuckGo rate flags
        return search.run(f"active job vacancies postings hiring in {user_region} {query}")

# 2. STREAMLIT WEB INTERFACE
st.set_page_config(page_title="AI Agent Career Strategist", layout="wide")
st.title("💼 AI Agent Career Strategist Network")
st.subheader("Optimize your resume using real-time market data")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    region = st.text_input("📍 Target Geographic Region", placeholder="e.g., New York, Remote US, Toronto", key="target_region")
with col2:
    timeframe = st.selectbox("⏳ Job Posting Timeframe", ["Past 24 hours", "Past week", "Past month"])

resume_method = st.radio("How would you like to provide your resume?", ["Paste Text", "Upload Word Document (.docx)"])
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

# 3. HIGH-QUALITY EXECUTION PIPELINE WITH POSITION INTELLIGENCE
def run_strategic_pipeline(selected_llm, sequential_fallback=False):
    max_words = 2000
    words = resume_content.split()
    safe_resume = " ".join(words[:max_words]) if len(words) > max_words else resume_content

    # Researcher Agent
    researcher = Agent(
        role='Market Research Analyst',
        goal=f'Locate raw search results of open job postings in {region} from the {timeframe}. Count the active instances and parse the exact titles.',
        backstory="A literal scraper analyst who extracts real-time metadata directly from raw search snippet blocks.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool], 
        llm=selected_llm
    )

    # Strategist Agent
    strategist = Agent(
        role='Executive Career Strategist',
        goal='Compile position metrics, identify technical skill gaps, and map comprehensive study paths.',
        backstory="Seasoned HR executive specializing in hiring market data interpretation and upskilling plans.",
        verbose=True,
        allow_delegation=False,
        llm=selected_llm
    )

    task_research = Task(
        description=f"""Execute an online search for active open vacancies in {region}. 
        Analyze the search results page carefully to derive:
        1. The count of relevant active positions surfaced in the snippets.
        2. The 5 most common specific position titles appearing in those active results.
        3. The top 5 technical skills required across those roles.
        
        CRITICAL: Do not provide historical assumptions. If you find data, list it explicitly.""",
        expected_output="A live market intelligence data report containing raw counts, explicit job titles found, and active skill requirements.",
        agent=researcher
    )

    task_strategy = Task(
        description=f"""Review this resume:
        ---
        {safe_resume}
        ---
        Using the live market research data report compiled by the researcher, format the final document. 
        You must explicitly display the structural position metrics (Total found and the 5 common names) right at the top, followed by the skill gap analysis and tailored learning milestones.""",
        expected_output="An executive optimization report containing a position market overview, missing qualifications, and learning milestones.",
        agent=strategist
    )

    # Multi-phase sequence for Groq or Single loop for Gemini
    if sequential_fallback:
        st.info("⚡ Phase 1: Analyzing Job Board Volumes & Position Names...")
        crew_1 = Crew(agents=[researcher], tasks=[task_research], process=Process.sequential)
        research_report = str(crew_1.kickoff())
        
        st.info("⚡ Phase 2: Mapping Resume Alignment & Recommendations...")
        task_strategy.description = f"Review this resume:\n---\n{safe_resume}\n---\nBased on this market data showing position volumes and titles:\n{research_report}\nFormulate the comprehensive strategy report."
        crew_2 = Crew(agents=[strategist], tasks=[task_strategy], process=Process.sequential)
        final_output = crew_2.kickoff()
        return final_output
    else:
        crew = Crew(agents=[researcher, strategist], tasks=[task_research, task_strategy], process=Process.sequential)
        return crew.kickoff()

# 4. EXECUTION FLOW
if st.button("🚀 Analyze Market & Enhance Resume"):
    if not resume_content or not region:
        st.error("Please provide both your resume and a target region.")
    else:
        with st.spinner("Processing analysis pipeline..."):
            try:
                if gemini_llm:
                    st.info("🔄 Querying Gemini 3.8 Flash...")
                    result = run_strategic_pipeline(gemini_llm, sequential_fallback=False)
                else:
                    raise Exception("429 Quota triggered.")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or not gemini_llm:
                    st.warning("⚠️ Primary limit reached. Running multi-phase fallback sequence to preserve output depth...")
                    if groq_llm:
                        try:
                            result = run_strategic_pipeline(groq_llm, sequential_fallback=True)
                        except Exception as groq_error:
                            st.error(f"Execution failed: {groq_error}")
                            st.stop()
                    else:
                        st.error("Missing backup API configurations.")
                        st.stop()
                else:
                    st.error(f"Unexpected error: {e}")
                    st.stop()

            st.success("✨ Analysis Complete!")
            st.markdown("### 📊 Market & Resume Strategy Report")
            st.markdown(result.raw if hasattr(result, 'raw') else str(result))
