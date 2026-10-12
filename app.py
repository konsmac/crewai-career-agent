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
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool # FIX: Using the core LangChain decorator compatible with 0.28.8

# 1. SETUP FREE LLM CONFIGURATION
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", # FIX: Targets the active flagship flash model 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

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

if not GOOGLE_API_KEY:
    st.warning("⚠️ GOOGLE_API_KEY environment variable is missing. Please add it to your Render dashboard settings.")

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

# 3. CREWAI AGENT ORCHESTRATION
if st.button("🚀 Analyze Market & Enhance Resume"):
    if not resume_content or not region:
        st.error("Please provide both your resume and a target region.")
    else:
        with st.spinner("Agents are actively researching live job boards and analyzing market data..."):
            
            # Agent 1: The Market Researcher
            researcher = Agent(
                role='Market Research Analyst',
                goal=f'Search and analyze live job postings in {region} from the {timeframe} matching the candidate profile.',
                backstory="An expert in finding trends on job boards. You identify what hiring managers are actively looking for right now.",
                verbose=True,
                allow_delegation=False,
                tools=[search_tool], 
                llm=llm
            )

            # Agent 2: The Resume & Career Strategist
            strategist = Agent(
                role='Executive Career Strategist',
                goal='Compare the user resume against live market demands and outline high-value skill gaps and study materials.',
                backstory="A seasoned HR executive who knows exactly how to make a resume stand out. You provide actionable upskilling paths.",
                verbose=True,
                allow_delegation=False,
                llm=llm
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
                {resume_content}
                ---
                Using the market research report compiled by the researcher, identify missing critical skills. 
                Provide a structured list of missing skills, specific certification recommendations, and free or premium study materials.""",
                expected_output="A tailored optimization strategy with structural resume advice and learning resources.",
                agent=strategist
            )

            # Run the Crew
            crew = Crew(
                agents=[researcher, strategist],
                tasks=[task_research, task_strategy],
                process=Process.sequential
            )
            
            try:
                result = crew.kickoff()
                
                # Output results to UI
                st.success("✨ Analysis Complete!")
                st.markdown("### 📊 Market & Resume Strategy Report")
                st.write(result.raw)
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
