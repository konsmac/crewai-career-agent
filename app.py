import streamlit as st
import os, docx2txt, asyncio
from crewai import Agent, Crew, Process, Task
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY

gemini_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3) if GOOGLE_API_KEY else None
groq_llm = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=GROQ_API_KEY, temperature=0.4, max_tokens=800) if GROQ_API_KEY else None

if GOOGLE_API_KEY: st.sidebar.success("🤖 Gemini Active")
elif GROQ_API_KEY: st.sidebar.warning("⚡ Groq Active")

@tool("Google Live Job Search Tool")
def search_tool(query: str) -> str:
    """Searches Google for real-time job vacancies matching the candidate profile."""
    if not os.environ.get("SERPER_API_KEY"): return "Error: SERPER_API_KEY missing."
    region = st.session_state.get('target_region', 'Montreal')
    resume = st.session_state.get('active_resume_text', '')
    p_context = resume[:250].replace('"', '').replace("'", "") if resume else ""
    clean_q = query.replace('"', '').replace("'", "")
    try:
        search = GoogleSerperAPIWrapper()
        oq = f"({clean_q}) {p_context[:80]} jobs openings hiring in {region} (site:://linkedin.com OR site:indeed.com)"
        return search.run(oq)
    except Exception as e: return f"Search error: {str(e)}"

st.set_page_config(page_title="AI Agent Career Strategist", layout="wide")
st.title("💼 AI Agent Career Strategist Network")
st.subheader("Optimize your resume using real-time market data")

col1, col2 = st.columns(2)
with col1: region = st.text_input("📍 Target Region", placeholder="e.g., Montreal", key="target_region", value="Montreal")
with col2: timeframe = st.selectbox("⏳ Timeframe", ["Past 24 hours", "Past week", "Past month"])

rm = st.radio("Provide resume via:", ["Paste Text", "Upload .docx"])
resume_content = ""
if rm == "Paste Text":
    resume_content = st.text_area("Ctrl+V your resume:", height=200)
    if resume_content: st.session_state['active_resume_text'] = resume_content
else:
    f = st.file_uploader("Upload Word file", type=["docx"])
    if f is not None:
        try:
            resume_content = docx2txt.process(f)
            st.session_state['active_resume_text'] = resume_content
            st.success("✅ Parsed successfully!")
        except Exception as e: st.error(f"File error: {e}")

def get_agents(llm):
    r = Agent(role='Market Analyst', goal=f'Find real job postings in {region} matching the candidate resume text context.', backstory="Parses target Google job indexing pools.", verbose=True, allow_delegation=False, tools=[search_tool], llm=llm)
    g = Agent(role='Gap Analyst', goal='Compare resume text against market vacancy findings to isolate skill flaws.', backstory="ATS keyword optimization consultant.", verbose=True, allow_delegation=False, llm=llm)
    e = Agent(role='Education Advisor', goal='Compile training roadmap milestones and learning paths.', backstory="HR upskilling specialist.", verbose=True, allow_delegation=False, llm=llm)
    return r, g, e

if st.button("🚀 Analyze Market & Enhance Resume"):
    res_txt = st.session_state.get('active_resume_text', '')
    if not res_txt or not region: st.error("Provide resume and region first.")
    else:
        ss, p1, p2, p3 = st.empty(), st.empty(), st.empty(), st.empty()
        safe_res = " ".join(res_txt.split()[:2000])
        with st.spinner("Executing pipeline..."):
            try:
                if gemini_llm:
                    ss.info("🔄 Querying Primary Engine...")
                    r, g, _ = get_agents(gemini_llm)
                    tr = Task(description=f"Search Google via tool for vacancies matching candidate skills in {region}. Count matching posts and find the top 5 titles.", expected_output="Metrics report.", agent=r)
                    ts = Task(description=f"Compare resume:\n{safe_res}\nagainst search data. Yield complete report.", expected_output="Full optimization analysis.", agent=g)
                    res = str(Crew(agents=[r, g], tasks=[tr, ts], process=Process.sequential).kickoff())
                    ss.success("✨ Complete!")
                    p1.markdown("### 📊 Market & Resume Strategy Report\n" + res)
                else: raise Exception("Fallback triggered.")
            except Exception as ex:
                if "429" in str(ex) or "quota" in str(ex).lower() or not gemini_llm:
                    ss.warning("⚠️ Limit reached. Initializing phased rendering pipeline...")
                    if groq_llm:
                        r, g, e = get_agents(groq_llm)
                        t1 = Task(description=f"Analyze snippets for {region} matching the candidate text profile domain. Return total matching counts and top 5 job titles.", expected_output="Table of matching positions.", agent=r)
                        ph1 = str(Crew(agents=[r], tasks=[t1], process=Process.sequential).kickoff())
                        p1.markdown(f"### 📊 1. Position Market Overview\n{ph1}")
                        t2 = Task(description=f"Compare resume:\n{safe_res}\nAgainst phase 1 data:\n{ph1}\nIsolate technical keyword deficiencies.", expected_output="List of skill gaps.", agent=g)
                        ph2 = str(Crew(agents=[g], tasks=[t2], process=Process.sequential).kickoff())
                        p2.markdown(f"### 🔍 2. Resume Gap Analysis\n{ph2}")
                        t3 = Task(description=f"Based on gaps:\n{ph2}\nRecommend certifications and training resources.", expected_output="Study path.", agent=e)
                        ph3 = str(Crew(agents=[e], tasks=[t3], process=Process.sequential).kickoff())
                        p3.markdown(f"### 🎓 3. Tailored Learning Path\n{ph3}")
                        ss.success("✨ All phases rendered perfectly!")
                    else: ss.error("❌ Missing backup API keys.")
                else: ss.error(f"Error: {ex}")
