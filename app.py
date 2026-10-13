import streamlit as st
import os, docx2txt, asyncio, requests, json

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from crewai import Agent, Crew, Process, Task
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool

# 1. SETUP ENVIRONMENTS & API HOOKS
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

gemini_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3) if GOOGLE_API_KEY else None
groq_llm = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=GROQ_API_KEY, temperature=0.4, max_tokens=800) if GROQ_API_KEY else None

if GOOGLE_API_KEY:
    st.sidebar.success("🤖 Primary Engine: Gemini 3.8 Flash Active")
elif GROQ_API_KEY:
    st.sidebar.warning("⚡ Backup Engine: Groq Qwen 3.8 Active")
if not SERPER_API_KEY:
    st.sidebar.error("⚠️ SERPER_API_KEY missing! Live Google data will fail.")

@tool("Google Live Job Search Tool")
def search_tool(query: str) -> str:
    """Searches Google for real-time live job postings, active hiring boards, and vacancy text snippets."""
    if not SERPER_API_KEY:
        return "Error: SERPER_API_KEY is missing."
    user_region = st.session_state.get('target_region', 'Montreal')
    url = "https://serper.dev"
    clean_query = query.replace('"', '').replace("'", "")
    payload = json.dumps({
        "q": f"{clean_query} jobs openings vacancies hiring in {user_region} (site:://linkedin.com OR site:indeed.com)",
        "num": 10
    })
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=12)
        if response.status_code != 200:
            return f"Error: Serper API server returned status code {response.status_code}."
        results = response.json()
        output_data = []
        if "organic" in results and results["organic"]:
            for item in results["organic"]:
                output_data.append(f"Title: {item.get('title', 'N/A')}\nSource: {item.get('link', 'N/A')}\nSnippet: {item.get('snippet', 'N/A')}\n---")
            return "\n".join(output_data)
        return f"No live search results came back from Google for '{clean_query}' in {user_region}."
    except Exception as e:
        return f"Network connectivity error trying to contact Google Search API: {str(e)}"

# 2. STREAMLIT WEB INTERFACE
st.set_page_config(page_title="AI Agent Career Strategist", layout="wide")
st.title("💼 AI Agent Career Strategist Network")
st.subheader("Optimize your resume using real-time market data")

col1, col2 = st.columns(2)
with col1:
    region = st.text_input("📍 Target Geographic Region", placeholder="e.g., Montreal, Toronto", key="target_region", value="Montreal")
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

    if use_split_pipeline:
        st.info("⚡ Phase 1/3: Querying Google API & Analyzing Live Vacancy Titles...")
        task_1 = Task(description=f"Analyze live search snippets for {region}. Review titles, count total openings found, and compile the 5 most frequent exact position names.", expected_output="A structured report displaying total positions and top 5 titles.", agent=researcher)
        phase_1_output = str(Crew(agents=[researcher], tasks=[task_1], process=Process.sequential).kickoff())

        st.info("⚡ Phase 2/3: Mapping Structural Keyword & Skill Gaps...")
        task_2 = Task(description=f"Compare this resume:\n---\n{safe_resume}\n---\nAgainst these openings:\n{phase_1_output}\nIdentify exactly which critical skills are missing.", expected_output="A structured list of core skill gaps and missing keywords.", agent=gap_analyst)
        phase_2_output = str(Crew(agents=[gap_analyst], tasks=[task_2], process=Process.sequential).kickoff())

        st.info("⚡ Phase 3/3: Assembling Actionable Education Roadmaps...")
        task_3 = Task(description=f"Based on these gaps:\n{phase_2_output}\nProvide high-value certification names and specific study materials.", expected_output="An actionable learning path with learning resources.", agent=education_advisor)
        phase_3_output = str(Crew(agents=[education_advisor], tasks=[task_3], process=Process.sequential).kickoff())

        return f"### 📊 1. Position Market Overview\n{phase_1_output}\n\n### 🔍 2. Resume Gap Analysis & ATS Evaluation\n{phase_2_output}\n\n### 🎓 3. Tailored Learning Path & Certifications\n{phase_3_output}"
    else:
        task_r = Task(description=f"Search Google via tool for active openings in {region}. Find the 5 most common position names and top 5 required skills.", expected_output="Live market data report with counts and job titles.", agent=researcher)
        task_s = Task(description=f"Review resume:\n---\n{safe_resume}\n---\nCompare it against research data to format a full gap report and study path.", expected_output="Full strategy report.", agent=gap_analyst)
        return str(Crew(agents=[researcher, gap_analyst], tasks=[task_r, task_s], process=Process.sequential).kickoff())

# 4. EXECUTION FLOW
if st.button("🚀 Analyze Market & Enhance Resume"):
    if not resume_content or not region:
        st.error("Please provide both your resume and a target region.")
    else:
        with st.spinner("Processing analysis pipeline..."):
            result = None
            try:
                if gemini_llm:
                    st.info("🔄 Querying Gemini 3.8 Flash...")
                    result = run_strategic_pipeline(gemini_llm, use_split_pipeline=False)
                else:
                    raise Exception("429 Quota triggered.")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or not gemini_llm:
                    st.warning("⚠️ Primary limit reached. Running fallback sequence...")
                    if groq_llm:
                        try:
                            result = run_strategic_pipeline(groq_llm, use_split_pipeline=True)
                        except Exception as groq_error:
                            st.error(f"Execution failed: {groq_error}")
                    else:
                        st.error("❌ Missing backup API configurations.")
                else:
                    st.error(f"Unexpected error: {error_msg}")

            if result:
                st.success("✨ Analysis Complete!")
                st.markdown("### 📊 Market & Resume Strategy Report")
                st.markdown(result)
            else:
                st.error("Pipeline finished but no output data was generated.")
