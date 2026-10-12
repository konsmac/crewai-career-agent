import streamlit as st
import docx2txt
import os
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit UI Configuration
st.set_page_config(page_title="AI Career Agent Network (Gemini)", page_icon="💼", layout="wide")

st.title("💼 AI Agent Network: Resume Optimizer & Job Market Analyst")
st.write("Powered by Google Gemini 1.5 Flash. Upload your resume, set your filters, and let the AI agents do the live research for you.")

# 1. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("🔑 API Keys Setup")
    gemini_api_key = st.text_input(
        "Google Gemini API Key", 
        type="password", 
        help="Get it free at ://google.com. It should start with 'AIzaSy'."
    )
    serper_api_key = st.text_input(
        "Serper API Key", 
        type="password", 
        help="Get it free at serper.dev for live Google search access."
    )
    
    st.header("⚙️ Job Search Filters")
    region = st.text_input("Geographic Region", placeholder="e.g., London, UK / Remote US")
    time_frame = st.selectbox(
        "Job Posting Timeframe",
        ["Past 24 hours", "Past week", "Past month", "Anytime"]
    )

# Stop app execution if keys are missing or invalid
if not gemini_api_key or not serper_api_key:
    st.warning("⚠️ Please provide both Google Gemini and Serper API Keys in the sidebar to unlock the application.")
    st.stop()

if not gemini_api_key.startswith("AIzaSy"):
    st.error("❌ The Gemini API Key format looks incorrect. Google Gemini keys must start with 'AIzaSy'. Please verify your key at Google AI Studio.")
    st.stop()

# Set environmental variables needed by the tools
os.environ["SERPER_API_KEY"] = serper_api_key

# 2. RESUME INPUT SECTION
st.subheader("📄 Your Resume / CV")
cv_option = st.radio("Choose input method:", ("Upload Word Document (.docx)", "Paste Raw Text (Ctrl+V)"))

resume_text = ""

if cv_option == "Upload Word Document (.docx)":
    uploaded_file = st.file_uploader("Choose a .docx file", type=["docx"])
    if uploaded_file is not None:
        resume_text = docx2txt.process(uploaded_file)
        st.success("✅ Resume file uploaded successfully!")
else:
    resume_text = st.text_area("Paste your resume text here:", height=250)

# 3. AGENT NETWORK EXECUTION
if st.button("🚀 Start Live Market Research"):
    if not resume_text.strip():
        st.error("❌ Please provide your resume content before running.")
    elif not region.strip():
        st.error("❌ Please specify a target geographic region in the sidebar.")
    else:
        with st.spinner("🕵️‍♂️ Gemini agents are analyzing live job markets... Please wait."):
            
            # Setup Google Gemini LLM & Search Tool
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=gemini_api_key,
                temperature=0.2
            )
            search_tool = SerperDevTool()

            # Define Agents
            labor_market_researcher = Agent(
                role='Senior Job Market Researcher',
                goal=f'Search and identify active job openings in {region} posted within the {time_frame} matching the resume profile.',
                backstory='An expert data miner specialized in tracking jobs boards and extracting real-time hiring metrics.',
                tools=[search_tool],
                llm=llm,
                verbose=True
            )

            career_strategist = Agent(
                role='Executive Recruiter & Career Coach',
                goal='Compare the user resume against market requirements to identify high-value skill gaps and suggest certifications.',
                backstory='A veteran talent acquisition advisor who knows exactly how to make profiles stand out to recruiters and ATS systems.',
                llm=llm,
                verbose=True
            )

            # Define Tasks
            research_task = Task(
                description=f"Analyze live job openings matching keywords found in this resume: \n\n{resume_text}\n\nFocus on {region} within the {time_frame}.",
                expected_output='A detailed research breakdown summarizing current job postings and in-demand skills.',
                agent=labor_market_researcher
            )

            strategy_task = Task(
                description="Review the researcher's findings against the original resume. 1) Identify high-value missing skills that make a candidate a top applicant. 2) Recommend specific online study materials or certifications to bridge those exact gaps.",
                expected_output='An actionable career roadmap covering profile compatibility, critical skill gaps, and certification/study recommendations.',
                agent=career_strategist
            )

            # Assemble & Run Crew
            career_crew = Crew(
                agents=[labor_market_researcher, career_strategist],
                tasks=[research_task, strategy_task],
                process=Process.sequential
            )

            result = career_crew.kickoff()

            st.success("✨ Optimization Analysis Completed!")
            st.markdown("### 📊 Your AI Agent Network Strategic Report")
            st.markdown(result.raw)
