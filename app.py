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

# Fallback engine (800 tokens per micro-phase is more than enough for individual sections)
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

# RE-ENGINEERED SEARCH TOOL WITH FOOTPRINT FORCING
@tool("Web Search Tool")
def search_tool(query: str) -> str:
    """Useful to search the internet for live job postings, market trends, and role vacancies."""
    search = DuckDuckGoSearchRun()
    user_region = st.session_state.get('target_region', 'Remote')
    optimized_query = f"{query} {user_region} site:://linkedin.com OR site:indeed.com"
    try:
        return search.run(optimized_query)
    except Exception:
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

# 3. ADVANCED 3-PHASE PIPELINE TO ELIMINATE OUTPUT TRUNCATION
def run_strategic_pipeline(selected_llm, use_split_pipeline=False):
    max_words = 2000
    words = resume_content.split()
    safe_resume = " ".join(words[:max_words]) if len(words) > max_words else resume_content

    # Defining the Core Specialized Agents
    researcher = Agent(
        role='Market Research Analyst',
        goal=f'Locate open job postings in {region} from the {timeframe}. Count active vacancies and parse exact titles.',
        backstory="A precise labor market trends scraper that extracts metrics from raw search snippets.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool], 
        llm=selected_llm
    )

    gap_analyst = Agent(
        role='Resume Gap Analyst',
        goal='Compare a technical resume against active market demands to uncover critical missing keywords and skills.',
        backstory="An expert ATS filtering specialist who exposes missing core assets in professional profiles.",
        verbose=True,
        allow_delegation=False,
        llm=selected_llm
    )

    education_advisor = Agent(
        role='Executive Education Consultant',
        goal='Design actionable study roadmaps, including real certifications and free learning materials.',
        backstory="A professional corporate learning curator who maps academic and skill development milestones.",
        verbose=True,
        allow_delegation=False,
        llm=selected_llm
    )

    # If Gemini hits a 429 quota block, execute using the 3-Phase Chain
    if use_split_pipeline:
        # PHASE 1: POSITION METRICS DISCOVERY
        st.info("⚡ Phase 1/3: Analyzing Job Board Volumes & Position Names...")
        task_1 = Task(
            description=f"Search for live vacancies matching this profile category in {region}. Count total openings found and extract the 5 most common exact titles.",
            expected_output="A structural markdown table featuring total counts and the top 5 job titles.",
            agent=researcher
        )
        crew_1 = Crew(agents=[researcher], tasks=[task_1], process=Process.sequential)
        phase_1_output = str(crew_1.kickoff())

        # PHASE 2: ATS RESUME GAP ANALYSIS
        st.info("⚡ Phase 2/3: Mapping Structural Keyword & Skill Gaps...")
        task_2 = Task(
            description=f"Compare this resume:\n---\n{safe_resume}\n---\nAgainst these discovered market needs:\n{phase_1_output}\nIdentify exactly which critical skills are missing.",
            expected_output="A structured list of core skill gaps and keywords missing from the applicant's profile.",
            agent=gap_analyst
        )
        crew_2 = Crew(agents=[gap_analyst], tasks=[task_2], process=Process.sequential)
        phase_2_output = str(crew_2.kickoff())

        # PHASE 3: UPSKILLING ROADMAP GENERATION
        st.info("⚡ Phase 3/3: Assembling Actionable Education Roadmaps...")
        task_3 = Task(
            description=f"Based on these identified skill gaps:\n{phase_2_output}\nProvide high-value certification names and specific study materials to fill them.",
            expected_output="An actionable learning path detailing training materials and certifications.",
            agent=education_advisor
        )
        crew_3 = Crew(agents=[education_advisor], tasks=[task_3], process=Process.sequential)
        phase_3_output = str(crew_3.kickoff())

        # Combine all separate chunks into a clean, complete report layout
        final_stitched_report = f"""
### 📊 1. Position Market Overview
{phase_1_output}

### 🔍 2. Resume Gap Analysis & ATS Evaluation
{phase_2_output}

### 🎓 3. Tailored Learning Path & Certifications
{phase_3_output}
"""
        return final_stitched_report

    else:
        # Standard workflow for Gemini (no truncation risk)
        task_r = Task(
            description=f"Search for active openings in {region}. Count instances and find the 5 most common position names and top 5 required skills.",
            expected_output="Market overview data.",
            agent=researcher
        )
        task_s = Task(
            description=f"Review resume:\n---\n{safe_resume}\n---\nCompare it against the research data to format a full gap report and study path.",
            expected_output="Full strategy report.",
            agent=gap_analyst
        )
        crew = Crew(agents=[researcher, gap_analyst], tasks=[task_r, task_s], process=Process.sequential)
        return str(crew.kickoff())

# 4. EXECUTION FLOW
if st.button("🚀 Analyze Market & Enhance Resume"):
    if not resume_content or not region:
        st.error("Please provide both your resume and a target region.")
    else:
        with st.spinner("Processing analysis pipeline..."):
            try:
                if gemini_llm:
                    st.info("🔄 Querying Gemini 3.8 Flash...")
                    result = run_strategic_pipeline(gemini_llm, use_split_pipeline=False)
                else:
                    raise Exception("429 Quota triggered.")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or not gemini_llm:
                    st.warning("⚠️ Primary limit reached. Running multi-phase fallback sequence to preserve output depth...")
                    if groq_llm:
                        try:
                            # Triggers the 3-Phase Chain to bypass the 800 token truncation window completely
                            result = run_strategic_pipeline(groq_llm, use_split_pipeline=True)
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
            st.markdown(result)
