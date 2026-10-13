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

# Fallback engine (using max_tokens=800 per task step to optimize memory window)
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

# WRAP THE DUCKDUCKGO TOOL
@tool("Web Search Tool")
def search_tool(query: str) -> str:
    """Useful to search the internet for live job postings, market trends, and role vacancies."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# 2. STREAMLIT WEB INTERFACE
st.set_page_config(page_title="AI Agent Career Strategist", layout="wide")
st.title("💼 AI Agent Career Strategist Network")
st.subheader("Optimize your resume using real-time market data")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    region = st.text_input("📍 Target Geographic Region", placeholder="e.g., New York, Remote US, Toronto")
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

# 3. HIGH-QUALITY EXECUTION PIPELINE
def run_strategic_pipeline(selected_llm, sequential_fallback=False):
    max_words = 2000
    words = resume_content.split()
    safe_resume = " ".join(words[:max_words]) if len(words) > max_words else resume_content

    # Researcher Agent
    researcher = Agent(
        role='Market Research Analyst',
        goal=f'Identify what hiring managers look for in {region} during the {timeframe}.',
        backstory="Expert job board trends miner.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool], 
        llm=selected_llm
    )

    # Strategist Agent
    strategist = Agent(
        role='Executive Career Strategist',
        goal='Identify technical skill gaps and map comprehensive study paths.',
        backstory="Seasoned HR executive specializing in high-value upskilling plans.",
        verbose=True,
        allow_delegation=False,
        llm=selected_llm
    )

    task_research = Task(
        description=f"Find live job vacancies in {region} for this profile category. Extract the top 5 technical skills required.",
        expected_output="A list of 5 key technical requirements discovered in active job ads.",
        agent=researcher
    )

    task_strategy = Task(
        description=f"Compare this resume:\n---\n{safe_resume}\n---\nAgainst the discovered skills. Output a gap analysis report.",
        expected_output="Comprehensive analysis outlining missing high-value qualifications and actionable learning milestones.",
        agent=strategist
    )

    # If Gemini failed previously, execute the tasks independently to reset the Output token window clock
    if sequential_fallback:
        st.info("⚡ Executing Market Discovery phase...")
        crew_1 = Crew(agents=[researcher], tasks=[task_research], process=Process.sequential)
        research_report = str(crew_1.kickoff())
        
        st.info("⚡ Executing Strategic Resume Alignment phase...")
        task_strategy.description = f"Review this resume:\n---\n{safe_resume}\n---\nBased on this market data:\n{research_report}\nFormulate a resume gap report."
        crew_2 = Crew(agents=[strategist], tasks=[task_strategy], process=Process.sequential)
        final_output = crew_2.kickoff()
        return final_output
    else:
        # Standard combined monolithic run for Gemini
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
