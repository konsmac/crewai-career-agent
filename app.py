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

# Initialize models carefully
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
) if GOOGLE_API_KEY else None

# FIXED: Swapped out the old string for Groq's active free-tier Llama 3.3 70B identifier
groq_llm = ChatGroq(
    model="llama-3.3-70b-specdec",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3
) if GROQ_API_KEY else None

# Sidebar status anchor
if GOOGLE_API_KEY:
    st.sidebar.success("🤖 Primary Engine: Gemini 3.8 Flash Active")
elif GROQ_API_KEY:
    st.sidebar.warning("⚡ Backup Engine: Groq Llama 3.3 Active (Gemini Skipped)")
else:
    st.error("⚠️ Missing API keys! Please provide GOOGLE_API_KEY or GROQ_API_KEY in Render configuration.")

# WRAP THE DUCKDUCKGO TOOL INSIDE A COMPATIBLE FUNCTION
@tool("Web Search Tool")
def search_tool(query: str) -> str:
    """Useful to search the internet for live job postings, market trends, and role vacancies."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# 2. STREAMLIT WEB INTERFACE
st.set_page_config(page_title="AI Agent Career Strategist", layout="wide")
st.title("💼 AI Agent Career Strategist Network")
st.subheader("Optimize your resume using real-time market data")

if not GOOGLE_API_KEY and not GROQ_API_KEY:
    st.warning("⚠️ Both GOOGLE_API_KEY and GROQ_API_KEY environment variables are missing. Please add at least one to your settings.")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    region = st.text_input("📍 Target Geographic Region", placeholder="e.g., New York, Remote US, Toronto")
with col2:
    timeframe = st.selectbox("⏳ Job Posting Timeframe", ["Past 24 hours", "Past week", "Past month"])

# Resume Input Options
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

# 3. CREWAI AGENT ORCHESTRATION FUNCTION WITH HARD FALLBACK EXCEPTION CATCHING
def run_job_crew(selected_llm):
    # DYNAMIC TPM COMPRESSION: Safeguards your free tier limits by cleaning and trimming unnecessary padding
    max_words = 2000
    words = resume_content.split()
    safe_resume = " ".join(words[:max_words]) if len(words) > max_words else resume_content

    # Agent 1: The Market Researcher
    researcher = Agent(
        role='Market Research Analyst',
        goal=f'Search and analyze live job postings in {region} from the {timeframe} matching the candidate profile.',
        backstory="An expert in finding trends on job boards. You identify what hiring managers are actively looking for right now.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool], 
        llm=selected_llm
    )

    # Agent 2: The Resume & Career Strategist
    strategist = Agent(
        role='Executive Career Strategist',
        goal='Compare the user resume against live market demands and outline high-value skill gaps and study materials.',
        backstory="A seasoned HR executive who knows exactly how to make a resume stand out. You provide actionable upskilling paths.",
        verbose=True,
        allow_delegation=False,
        llm=selected_llm
    )

    # Task 1: Research the market
    task_research = Task(
        description=f"""Use the search tool to find live job vacancies or job market criteria relevant to the user profile concept.
        Target Region: {region}
        Timeframe constraint: {timeframe}
        Identify the top 5 highly repeated keywords and technical expectations in recent postings.""",
        expected_output="A structured report listing top trending skills and core requirements based on live roles.",
        agent=researcher
    )

    # Task 2: Gap Analysis and Upskilling Recommendations
    task_strategy = Task(
        description=f"""Review this user resume:
        ---
        {safe_resume}
        ---
        Using the market research report compiled by the researcher, identify missing critical skills. 
        Provide a structured list of missing skills, specific certification recommendations, and free or premium study materials.""",
        expected_output="A tailored optimization strategy with structural resume advice and learning resources.",
        agent=strategist
    )

    crew = Crew(
        agents=[researcher, strategist],
        tasks=[task_research, task_strategy],
        process=Process.sequential
    )
    
    return crew.kickoff()

# 4. EXECUTION FLOW
if st.button("🚀 Analyze Market & Enhance Resume"):
    if not gemini_llm and not groq_llm:
        st.error("No valid LLM models configured. Check your API Keys.")
    elif not resume_content or not region:
        st.error("Please provide both your resume and a target region.")
    else:
        with st.spinner("Agents are actively researching live job boards..."):
            # Attempt execution via primary engine (Gemini) first
            try:
                if gemini_llm:
                    st.info("🔄 Querying Gemini 3.8 Flash...")
                    result = run_job_crew(gemini_llm)
                else:
                    raise Exception("Gemini key missing, falling back.")
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or not gemini_llm:
                    st.warning("⚠️ Gemini Free Quota Exceeded (429)! Instantly routing agents to Groq Llama 3.3 fallback engine...")
                    
                    if groq_llm:
                        try:
                            result = run_job_crew(groq_llm)
                        except Exception as groq_error:
                            st.error(f"Groq Engine execution error: {groq_error}")
                            st.stop()
                    else:
                        st.error("❌ Gemini failed and no GROQ_API_KEY was found in environment to recover execution.")
                        st.stop()
                else:
                    st.error(f"An unexpected error occurred: {e}")
                    st.stop()

            # Output results to UI safely
            st.success("✨ Analysis Complete!")
            st.markdown("### 📊 Market & Resume Strategy Report")
            
            if hasattr(result, 'raw'):
                st.markdown(result.raw)
            else:
                st.markdown(str(result))
