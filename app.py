import streamlit as st
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# OpenAI API 키 초기화
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
if not OPENAI_API_KEY:
    st.error("OpenAI API key가 설정되어 있지 않습니다. 환경변수를 확인해주세요.")
    st.stop()

# 상담 시나리오 (내부 프롬프트용 – 출력에는 포함되지 않음)
CONSULTATION_SCENARIO = """
1. 상담 시작 – 신뢰 형성
안녕하세요, ○○ 어머님(아버님). 만나 뵙게 되어 반갑습니다. 저는 ○○반 담임 △△△ 교사입니다.
오늘 상담을 통해 ○○이가 학교에서 잘 적응할 수 있도록 학부모님과 이야기를 나누고자 합니다.
학기 초라서 아직 모든 학생을 깊이 파악하지는 못했지만, 학부모님께서 ○○이에 대해 알려주시면 학교에서도 더 잘 지도할 수 있을 것 같습니다.
   - 학부모와의 협력을 강조하며 신뢰를 형성합니다.
   - 교사가 학생을 완벽히 파악하지 못한 상태임을 자연스럽게 전달합니다.

2. 학부모 의견을 먼저 듣기
○○이는 집에서 어떤 성향인가요?
평소에 조용한 편인지, 활동적인 편인지, 관심 있는 활동이 있는지 궁금합니다.
또 ○○이가 학교에서 잘 적응하려면 어떤 부분을 지원해 주면 좋을까요?
   - 부모가 먼저 아이의 성향을 설명할 수 있도록 유도합니다.
   - 부모의 설명을 통해 교사가 파악하지 못한 정보를 얻을 수 있습니다.

3. 현재까지의 학교생활 공유 및 협력 방안 논의
○○이는 학교에서 점점 적응해 나가고 있는 모습입니다.
아직은 (예: 새로운 친구들과 친해지는 과정 중이며, 본인의 의견을 표현하는 방식도 탐색 중입니다).
학기 초라 모든 학생들이 서로 알아가는 단계라 앞으로 어떤 모습을 보여줄지 기대됩니다.
혹시 ○○이가 학교에서 잘 적응할 수 있도록 학부모님께서 특별히 신경 써 주셨으면 하는 부분이 있을까요?
   - 교사의 중립적인 관찰 내용을 공유하면서 학부모의 의견을 들을 수 있는 여지를 남깁니다.
   - 학부모의 기대나 우려 사항을 파악하여 협력 방안을 조율할 수 있습니다.

4. 상담 마무리 – 지속적인 협력 강조
○○이가 학교에서 즐겁게 생활하고 성장할 수 있도록 계속 관심을 가지고 지도하겠습니다.
혹시 학교생활과 관련해 궁금한 점이나 고민되는 부분이 있으시면 언제든 연락 주십시오.
가정과 학교가 함께 협력할 때 아이들은 더욱 건강하게 성장할 수 있습니다.
   - 상담이 일회성으로 끝나지 않도록 지속적인 소통 의지를 강조합니다.
   - 가정과 학교가 함께 아이에게 관심을 가져야 한다는 점을 안내합니다.

※ 인사는 최초 인사 시에만 사용하고, 이후에는 반복하지 마세요.
"""

def set_page_config():
    try:
        st.set_page_config(
            page_title="학부모 상담 채팅", 
            page_icon="👨‍👩‍👧‍👦", 
            layout="wide"
        )
    except Exception as e:
        st.error(f"페이지 설정 오류: {e}")
    st.markdown(
        """
        <style>
        .main .block-container {
            padding: 2rem;
            max-width: 1200px;
            font-size: 1rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def generate_system_prompt(data):
    prompt = f"""다음 상담 정보를 바탕으로 상담을 진행하세요.

[상담 정보]
- 학교급: {data.get('school_type', '')}
- 성별: {data.get('gender', '')}
- 학년: {data.get('grade', '')}
- 주요 상담 내용: {data.get('counseling_issue', '')}
"""
    return prompt

def generate_role_system_prompt(role, data):
    base_prompt = generate_system_prompt(data)
    if role == "선생님 -> 가상 학부모":
        role_prompt = (
            "당신은 인격과 개성이 뚜렷한 가상 학부모입니다. 선생님이 제공한 정보를 바탕으로, "
            "이전 대화 내용을 기억하며 자연스럽고 연속적인 대화를 진행하세요. 자녀의 학교생활과 가정생활에 관한 고민이나 질문에 대해 진솔하고 구체적으로 답변하세요."
        )
    elif role == "학부모 -> 가상 선생님":
        role_prompt = (
            "당신은 인격이 부여된 따뜻한 성품의 경험 많은 선생님입니다. 학부모의 메시지를 바탕으로, "
            "이전 대화 맥락과 연결하여 자연스러운 대화를 이어가며 자녀의 학교생활 및 가정생활에 관한 고민과 질문에 대해 공감과 조언을 포함한 답변을 작성하세요."
        )
    elif role == "학생 -> 가상 선생님":
        role_prompt = (
            "당신은 인격과 따뜻함이 있는 선생님입니다. 학생의 메시지를 바탕으로, 이전 대화 내용과 연결하여 명확하고 친근하게 답변을 제공하세요."
        )
    elif role == "선생님 -> 가상 학생":
        role_prompt = (
            "당신은 인격이 부여된 학생입니다. 선생님이 제공한 정보를 참고하여, 이전 대화 내용을 반영한 자연스러운 흐름의 대화를 진행하며 솔직하고 진솔한 답변을 작성하세요."
        )
    else:
        role_prompt = ""
    prompt = base_prompt + "\n" + role_prompt
    if role in ["선생님 -> 가상 학부모", "학부모 -> 가상 선생님"]:
        prompt += "\n\n[상담 시나리오 참고]\n" + CONSULTATION_SCENARIO
    return prompt

def initialize_chat_history(data, role):
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        system_prompt = generate_role_system_prompt(role, data)
        st.session_state.chat_history = [{"role": "system", "content": system_prompt}]
    if "greeting_sent" not in st.session_state:
        st.session_state.greeting_sent = False

def summarize_chat_history(messages, max_tokens=150):
    conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    summarization_prompt = (
        "다음 대화 내용을 간결하게 요약해줘. "
        "핵심 포인트와 중요한 정보를 포함하되, 불필요한 세부 사항은 생략하고 자연스럽게 연결되는 요약문을 작성해줘.\n\n"
        f"{conversation_text}"
    )
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",
        temperature=0.5,
        max_tokens=max_tokens
    )
    response = chat.invoke([{"role": "system", "content": summarization_prompt}])
    return response.content.strip()

def get_recent_context(chat_history, max_messages=6):
    if len(chat_history) <= max_messages + 1:
        return chat_history
    else:
        system_message = chat_history[0]
        messages_to_summarize = chat_history[1:-max_messages]
        summary = summarize_chat_history(messages_to_summarize)
        summary_message = {"role": "system", "content": f"이전 대화 요약: {summary}"}
        return [system_message, summary_message] + chat_history[-max_messages:]

def generate_closing_message(role, chat_history):
    closing_instruction = (
        "대화를 마무리하는 말로, 오늘 상담에 참여해 주셔서 감사합니다. "
        "앞으로도 지속적으로 소통하며 도와드리겠습니다. 좋은 하루 보내세요."
    )
    if role == "선생님 -> 가상 학부모":
        closing_instruction = "안녕히 계세요. 오늘 상담에 참여해 주셔서 감사합니다. 앞으로도 지속적으로 소통하며 도와드리겠습니다. 좋은 하루 보내세요."
    elif role == "학부모 -> 가상 선생님":
        closing_instruction = "감사합니다. 오늘 상담을 통해 많은 도움이 되셨길 바랍니다. 앞으로도 계속 연락 드리겠습니다. 좋은 하루 되세요."
    elif role == "학생 -> 가상 선생님":
        closing_instruction = "오늘 상담해 주셔서 감사합니다. 앞으로도 도움이 필요하면 언제든 말씀해주세요. 안녕히 계세요."
    elif role == "선생님 -> 가상 학생":
        closing_instruction = "잘 들었습니다. 오늘 상담에 참여해 주셔서 감사합니다. 좋은 하루 보내세요."
    
    closing_prompt = (
        f"이전 대화 내용을 참고하여, 자연스럽게 마무리하는 인사말을 작성해줘. 다음 문장을 참고해:\n{closing_instruction}"
    )
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",
        temperature=0.5,
        max_tokens=150
    )
    response = chat.invoke([{"role": "system", "content": closing_prompt}])
    return response.content.strip()

# 각 역할별 응답 생성 시 한 메시지에는 하나의 질문과 하나의 주제만 포함하도록 지시
def generate_parent_response(chat_history):
    greeting_line = ""
    if not st.session_state.get("greeting_sent", False):
        greeting_line = "안녕하세요, 만나 뵙게 되어 반갑습니다. "
        st.session_state.greeting_sent = True
    parent_instruction = (
        greeting_line +
        "당신은 인격과 개성이 뚜렷한 가상 학부모입니다. 선생님이 최근에 언급한 내용을 포함해, "
        "자녀의 학교생활이나 가정생활에 관한 구체적인 고민, 질문, 의견을 진솔하게 작성하세요. "
        "이전 대화 내용과 연결해서 자연스럽게 이어가 주세요. "
        "한 번의 메시지에는 하나의 질문과 하나의 내용에 대해서만 이야기해 주세요."
    )
    history = chat_history + [{"role": "system", "content": parent_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",
        temperature=0.6,
        max_tokens=800
    )
    response = chat.invoke(recent_history)
    return response.content.strip()

def generate_teacher_response(chat_history):
    greeting_line = ""
    if not st.session_state.get("greeting_sent", False):
        greeting_line = "안녕하세요, 만나 뵙게 되어 반갑습니다. "
        st.session_state.greeting_sent = True
    teacher_instruction = (
        greeting_line +
        "당신은 인격이 부여된 따뜻하고 경험 많은 선생님입니다. "
        "이전 대화 내용을 바탕으로 대화의 흐름을 유지하며, 학부모나 학생의 최신 메시지에 대해 "
        "자녀의 학교생활과 가정생활에 관한 고민, 질문, 의견에 공감하고 구체적인 조언을 포함한 답변을 작성하세요. "
        "이전 대화 내용과 연결해서 자연스럽게 이어가 주세요. "
        "한 번의 메시지에는 하나의 질문과 하나의 주제에 대해서만 답변해 주세요."
    )
    history = chat_history + [{"role": "system", "content": teacher_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=800
    )
    response = chat.invoke(recent_history)
    return response.content.strip()

def generate_student_response(chat_history):
    greeting_line = ""
    if not st.session_state.get("greeting_sent", False):
        greeting_line = "안녕하세요, 만나 뵙게 되어 반갑습니다. "
        st.session_state.greeting_sent = True
    student_instruction = (
        greeting_line +
        "당신은 인격이 부여된 학생으로, 선생님의 최신 메시지에 대해 자연스럽게 연결되는 질문이나 의견을 진솔하게 작성하세요. "
        "이전 대화 내용과 연결해서 자연스럽게 이어가 주세요. "
        "한 번의 메시지에는 하나의 질문과 하나의 주제에 대해서만 이야기해 주세요."
    )
    history = chat_history + [{"role": "system", "content": student_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o",
        temperature=0.6,
        max_tokens=800
    )
    response = chat.invoke(recent_history)
    return response.content.strip()

def generate_teacher_input_suggestions(chat_history):
    suggestion_instruction = (
        "당신은 인격과 경험이 풍부한 선생님입니다. 지금까지의 대화 내용을 반영하여, "
        "가상 학부모에게 전달할 추천 대화 예시 3가지를 아래 형식에 맞추어 제시해주세요. "
        "출력 시 상담 정보는 포함하지 말고, 각 예시는 '예시 A:', '예시 B:', '예시 C:'로 구분하여 작성하세요.\n\n"
        "【예시 대화 형식】\n"
        "예시 A: [대화 예시 내용]\n"
        "예시 B: [대화 예시 내용]\n"
        "예시 C: [대화 예시 내용]\n\n"
        "상담 정보는 내부적으로만 반영하고, 출력에는 포함하지 마세요."
    )
    history = chat_history + [{"role": "system", "content": suggestion_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
         openai_api_key=OPENAI_API_KEY,
         model="gpt-4o",
         temperature=0.7,
         max_tokens=800
    )
    response = chat.invoke(recent_history)
    suggestions_text = response.content.strip()
    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
    return suggestions

def generate_parent_input_suggestions(chat_history):
    suggestion_instruction = (
        "당신은 인격이 부여된 따뜻한 학부모입니다. 지금까지의 대화 내용을 반영하여, "
        "가상 선생님에게 전달할 추천 대화 예시 3가지를 아래 형식에 맞추어 제시해주세요. "
        "출력 시 상담 정보는 포함하지 말고, 각 예시는 '예시 A:', '예시 B:', '예시 C:'로 구분하여 작성하세요.\n\n"
        "【예시 대화 형식】\n"
        "예시 A: [대화 예시 내용]\n"
        "예시 B: [대화 예시 내용]\n"
        "예시 C: [대화 예시 내용]\n\n"
        "상담 정보는 내부적으로만 반영하고, 출력에는 포함하지 마세요."
    )
    history = chat_history + [{"role": "system", "content": suggestion_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
         openai_api_key=OPENAI_API_KEY,
         model="gpt-4o",
         temperature=0.7,
         max_tokens=800
    )
    response = chat.invoke(recent_history)
    suggestions_text = response.content.strip()
    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
    return suggestions

def generate_student_input_suggestions(chat_history):
    suggestion_instruction = (
        "당신은 인격이 부여된 학생입니다. 지금까지의 대화 내용을 반영하여, "
        "가상 선생님에게 전달할 추천 대화 예시 3가지를 아래 형식에 맞추어 제시해주세요. "
        "출력 시 상담 정보는 포함하지 말고, 각 예시는 '예시 A:', '예시 B:', '예시 C:'로 구분하여 작성하세요.\n\n"
        "【예시 대화 형식】\n"
        "예시 A: [대화 예시 내용]\n"
        "예시 B: [대화 예시 내용]\n"
        "예시 C: [대화 예시 내용]\n\n"
        "상담 정보는 내부적으로만 반영하고, 출력에는 포함하지 마세요."
    )
    history = chat_history + [{"role": "system", "content": suggestion_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
         openai_api_key=OPENAI_API_KEY,
         model="gpt-4o",
         temperature=0.7,
         max_tokens=800
    )
    response = chat.invoke(recent_history)
    suggestions_text = response.content.strip()
    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
    return suggestions

def generate_teacher_suggestions(chat_history):
    suggestion_instruction = (
        "당신은 인격이 부여된 경험 많은 선생님입니다. 지금까지의 대화 내용을 반영하여, "
        "학부모나 학생의 최신 메시지를 기반으로 한 추천 대화 예시 3가지를 아래 형식에 맞추어 제시해주세요. "
        "출력 시 상담 정보는 포함하지 말고, 각 예시는 '예시 A:', '예시 B:', '예시 C:'로 구분하여 작성하세요.\n\n"
        "【예시 대화 형식】\n"
        "예시 A: [대화 예시 내용]\n"
        "예시 B: [대화 예시 내용]\n"
        "예시 C: [대화 예시 내용]\n\n"
        "상담 정보는 내부적으로만 반영하고, 출력에는 포함하지 마세요."
    )
    history = chat_history + [{"role": "system", "content": suggestion_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
         openai_api_key=OPENAI_API_KEY,
         model="gpt-4o",
         temperature=0.7,
         max_tokens=800
    )
    response = chat.invoke(recent_history)
    suggestions_text = response.content.strip()
    suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
    return suggestions

def main():
    set_page_config()
    
    # 세션 상태 초기화
    if "data" not in st.session_state:
        st.session_state.data = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.markdown(
        "<div style='text-align:center'><h1>👨‍👩‍👧‍👦 학부모 상담 채팅</h1>"
        "<p>대화 역할을 선택하여 메시지를 입력하면, 인격과 개성이 반영된 상대방이 이전 대화 맥락과 상담 정보를 기억하며 자연스럽게 대화를 이어갑니다.<br>"
        "예: <b>선생님 -> 가상 학부모</b>, <b>학부모 -> 가상 선생님</b>, <b>학생 -> 가상 선생님</b>, <b>선생님 -> 가상 학생</b></p></div>",
        unsafe_allow_html=True,
    )
    
    st.sidebar.markdown("## 상담 정보 입력")
    with st.sidebar.form("info_form"):
        school_type = st.selectbox("학교급", ["초등학교", "중학교"])
        gender = st.selectbox("성별", ["남성", "여성"])
        grade_options = (["1학년", "2학년", "3학년", "4학년", "5학년", "6학년"]
                         if school_type == "초등학교" else ["1학년", "2학년", "3학년"])
        grade = st.selectbox("학년", grade_options)
        counseling_issue = st.text_area("상담할 주요 내용", placeholder="예) 학교 생활, 친구 관계, 학업 부담 등", height=100)
        submit_info = st.form_submit_button("상담 정보 저장")
    
    if submit_info:
        st.session_state.data = {
            "school_type": school_type,
            "gender": gender,
            "grade": grade,
            "counseling_issue": counseling_issue,
            "consultation_date": datetime.now().strftime("%Y-%m-%d")
        }
        st.session_state.chat_history = []
        st.session_state.greeting_sent = False
        st.sidebar.success("상담 정보가 저장되었습니다. 중앙 채팅창에서 대화를 진행하세요.")
    
    mode_config = {
        "선생님 -> 가상 학부모": {"input_avatar": "👨‍🏫", "response_avatar": "👨‍👩‍👧‍👦"},
        "학부모 -> 가상 선생님": {"input_avatar": "👨‍👩‍👧‍👦", "response_avatar": "👨‍🏫"},
        "학생 -> 가상 선생님": {"input_avatar": "🧑‍🎓", "response_avatar": "👨‍🏫"},
        "선생님 -> 가상 학생": {"input_avatar": "👨‍🏫", "response_avatar": "🧑‍🎓"}
    }
    role_mode = st.selectbox("대화 역할 선택", list(mode_config.keys()))
    
    if not st.session_state.chat_history:
        if st.session_state.data:
            initialize_chat_history(st.session_state.data, role_mode)
    
    st.markdown("## 상담 채팅")
    if st.session_state.chat_history:
        for message in st.session_state.chat_history[1:]:
            msg_mode = message.get("mode", role_mode)
            if message["role"] == "assistant":
                avatar = mode_config[msg_mode]["response_avatar"]
                st.chat_message("assistant", avatar=avatar).write(message["content"])
            elif message["role"] == "user":
                avatar = mode_config[msg_mode]["input_avatar"]
                st.chat_message("user", avatar=avatar).write(message["content"])
    
    user_input = st.chat_input("메시지를 입력하세요")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input, "mode": role_mode})
        if role_mode == "선생님 -> 가상 학부모":
            with st.spinner("가상 학부모 응답 생성 중..."):
                reply = generate_parent_response(st.session_state.chat_history)
        elif role_mode in ["학부모 -> 가상 선생님", "학생 -> 가상 선생님"]:
            with st.spinner("가상 선생님 응답 생성 중..."):
                reply = generate_teacher_response(st.session_state.chat_history)
        elif role_mode == "선생님 -> 가상 학생":
            with st.spinner("가상 학생 응답 생성 중..."):
                reply = generate_student_response(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": reply, "mode": role_mode})
        st.rerun()
    
    if st.button("추천 답변 보기"):
        if role_mode == "선생님 -> 가상 학부모":
            suggestions = generate_teacher_input_suggestions(st.session_state.chat_history)
        elif role_mode == "학부모 -> 가상 선생님":
            suggestions = generate_parent_input_suggestions(st.session_state.chat_history)
        elif role_mode == "학생 -> 가상 선생님":
            suggestions = generate_student_input_suggestions(st.session_state.chat_history)
        elif role_mode == "선생님 -> 가상 학생":
            suggestions = generate_teacher_suggestions(st.session_state.chat_history)
        st.session_state.teacher_suggestions = suggestions
    
    if "teacher_suggestions" in st.session_state:
        st.markdown("### 추천 대화 예시")
        for suggestion in st.session_state.teacher_suggestions:
            st.write(suggestion)
    
    if st.button("대화 종료"):
        with st.spinner("대화 마무리 메시지 생성 중..."):
            closing_reply = generate_closing_message(role_mode, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": closing_reply, "mode": role_mode})
        st.rerun()
    
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        if "teacher_suggestions" in st.session_state:
            del st.session_state.teacher_suggestions
        st.session_state.greeting_sent = False
        st.rerun()

if __name__ == "__main__":
    main()

