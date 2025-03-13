import streamlit as st
import time
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ------------------------
# 기존 코드에서 사용하던 상수, 딕셔너리, 함수들
# ------------------------

CONVERSATION_LENGTH_GUIDELINES = {
    "단계 1 (짧음)": "간결하고 핵심적인 내용만 담아 짧게 응답하세요. 1-2개의 핵심 포인트만 언급하고, 상세한 설명은 생략하세요. 대략 100-150단어 정도로 답변을 제한하세요.",
    "단계 2 (보통)": "균형 잡힌 길이로 응답하세요. 2-3개의 주요 포인트를 포함하고, 각 포인트에 대한 기본적인 설명을 제공하세요. 대략 200-300단어 정도가 적절합니다.",
    "단계 3 (긴 대화)": "더 상세하고 포괄적인 응답을 제공하세요. 3-4개의 주요 포인트를 다루고, 각 포인트에 대한 자세한 설명과 예시를 포함하세요. 필요한 경우 추가 정보나 관련 내용도 다룰 수 있습니다. 대략 350-500단어 정도가 적절합니다."
}

# Big Five 성격 특성 정보 (기존: BIG_FIVE_TRAITS)
BIG_FIVE_TRAITS = {
    # 개방성 (Openness)
    "높은 개방성": "새로운 경험, 아이디어, 창의적 사고에 대한 개방성이 높습니다. 상상력이 풍부하고 호기심이 많으며, 예술적 감각과 지적 탐구를 즐깁니다. 다양한 학습 방법에 적응하며 특히 창의적이고 혁신적인 접근 방식에 흥미를 느낍니다.",
    "중간 개방성": "새로운 아이디어와 전통적인 방식 사이에서 균형을 유지합니다. 필요에 따라 혁신적인 방법을 시도하면서도 검증된 방식의 가치를 인정합니다. 다양한 학습 환경에 적응하며 실용적인 창의성을 발휘합니다.",
    "낮은 개방성": "전통적이고 검증된 방식을 선호하며 실용적인 접근법을 중시합니다. 구체적인 지시와 명확한 구조가 있는 학습 환경에서 편안함을 느낍니다. 일관성과 안정성을 추구하며 이론보다 실제 적용 가능한 지식에 집중합니다.",

    # 성실성 (Conscientiousness)
    "높은 성실성": "조직적이고 계획적이며 목표 지향적입니다. 자기 규율이 뛰어나고 책임감이 강하며, 세부 사항에 주의를 기울입니다. 체계적인 학습 계획을 세우고 과제를 철저히 완수하며, 높은 학업 성취를 위해 꾸준히 노력합니다.",
    "중간 성실성": "적절한 계획성과 유연성을 모두 갖추고 있습니다. 중요한 일에 집중하면서도 상황에 따라 계획을 조정할 수 있습니다. 과제 완수를 중요시하지만, 완벽주의에 빠지지 않으며 균형 잡힌 접근법을 취합니다.",
    "낮은 성실성": "자유롭고 유연한 접근 방식을 선호하며 즉흥적인 성향이 있습니다. 엄격한 일정이나 세부 규칙보다는 자유로운 탐색을 통해 학습합니다. 다양한 활동을 동시에 진행하는 경향이 있으며, 창의적 문제 해결에 능할 수 있습니다.",

    # 외향성 (Extraversion)
    "높은 외향성": "사교적이고 활동적이며 에너지가 넘칩니다. 그룹 활동과 토론을 통해 학습하는 것을 즐기며, 다른 사람들과의 상호작용에서 에너지를 얻습니다. 발표와 공동 작업 같은 활동적인 학습 환경에서 뛰어난 성과를 보입니다.",
    "중간 외향성": "사회적 상황과 독립적인 활동 사이에서 균형을 유지합니다. 그룹 활동에 참여하는 것을 즐기면서도 혼자만의 시간을 통해 재충전할 수 있습니다. 상황에 따라 적극적으로 참여하거나 조용히 관찰하는 유연성을 갖추고 있습니다.",
    "낮은 외향성": "조용하고 독립적인 활동을 선호하며 깊은 사고와 개인적 성찰을 중시합니다. 혼자 학습하는 것을 편안해하며, 소규모 그룹이나 일대일 상호작용에서 더 적극적으로 참여합니다. 내면의 생각과 아이디어를 발전시키는 데 능숙합니다.",

    # 친화성 (Agreeableness)
    "높은 친화성": "협조적이고 공감 능력이 뛰어나며 타인을 배려합니다. 조화로운 관계를 중시하고 갈등을 피하는 경향이 있습니다. 협력 학습에 적극적으로 참여하며, 다른 학생들을 돕고 지원하는 역할을 자연스럽게 맡습니다.",
    "중간 친화성": "협력과 개인적 목표 사이에서 균형을 유지합니다. 다른 사람들과 잘 협력하면서도 필요할 때는 자신의 의견을 주장할 수 있습니다. 팀 활동에서 조화를 추구하면서도 공정성과 개인적 원칙을 중요시합니다.",
    "낮은 친화성": "독립적인 사고와 직설적인 의사소통 방식을 가지고 있습니다. 타인의 의견보다 논리와 원칙에 따라 결정을 내리며, 비판적 사고에 능숙합니다. 경쟁적인 환경에서도 편안함을 느끼며, 솔직한 피드백을 주고받는 것을 선호합니다.",

    # 정서적 안정성 / 신경증 (Neuroticism)
    "높은 정서적 안정성": "스트레스와 압박 상황에서도 감정적으로 안정적인 모습을 보입니다. 실패를 학습 기회로 받아들이고 회복력이 뛰어납니다. 어려운 과제에도 침착하게 접근하며, 불확실성을 효과적으로 관리합니다.",
    "중간 정서적 안정성": "감정의 균형을 잘 유지하며 적절한 정서적 반응을 보입니다. 중요한 상황에서는 적당한 긴장감을 느끼지만, 이를 효과적으로 관리할 수 있습니다. 스트레스 요인을 인식하고 대응하는 능력이 있습니다.",
    "높은 신경증": "감정 변화가 크고 스트레스에 민감하게 반응합니다. 실수나 비판에 대해 걱정하는 경향이 있으며, 불안감이 학습에 영향을 줄 수 있습니다. 명확한 기대치와 정기적인 피드백이 있는 구조화된 환경에서 더 잘 학습합니다."
}

# MBTI 데이터
MBTI_CHARACTERISTICS = {
    # 직관형(N) + 사고형(T) 조합
    "INTJ": "분석적이고 논리적이며 전략적 사고를 중시합니다. 지식 습득에 열정적이며 독립적입니다.",
    "INTP": "지적 호기심이 많고 논리적으로 복잡한 문제를 해결하는 것을 좋아합니다. 독립적이고 창의적입니다.",
    "ENTJ": "리더십이 강하고 효율성을 중시합니다. 목표 지향적이며 직설적인 소통 방식을 선호합니다.",
    "ENTP": "창의적이고 논쟁을 즐기며 새로운 아이디어에 열정적입니다. 호기심이 많고 적응력이 뛰어납니다.",

    # 직관형(N) + 감정형(F) 조합
    "INFJ": "이상주의적이고 통찰력이 뛰어납니다. 깊은 관계를 중시하며 타인의 성장을 돕는 것에 관심이 많습니다.",
    "INFP": "이상주의적이고 창의적이며 자신의 가치와 신념에 충실합니다. 공감 능력이 뛰어나고 개인의 고유성을 중시합니다.",
    "ENFJ": "사람들을 이끌고 영감을 주는 능력이 있습니다. 타인의 필요를 민감하게 감지하고 조화를 추구합니다.",
    "ENFP": "열정적이고 창의적이며 가능성을 발견하는 능력이 뛰어납니다. 사람과의 관계를 중시하고 적응력이 뛰어납니다.",

    # 감각형(S) + 사고형(T) 조합
    "ISTJ": "책임감이 강하고 체계적이며 사실에 기반한 판단을 합니다. 규칙과 전통을 존중하고 신뢰성을 중시합니다.",
    "ISTP": "문제 해결 능력이 뛰어나고 실용적입니다. 독립적이며 위기 상황에서 침착하게 대응합니다.",
    "ESTJ": "체계적이고 실용적이며 결과 지향적입니다. 명확한 규칙과 절차를 중시하고 직접적인 소통을 선호합니다.",
    "ESTP": "즉흥적이고 활동적이며 실용주의적입니다. 현재에 집중하고 문제 해결에 직접적으로 접근합니다.",

    # 감각형(S) + 감정형(F) 조합
    "ISFJ": "세심하고 헌신적이며 책임감이 강합니다. 다른 사람의 필요를 세심하게 살피고 전통과 안정을 중시합니다.",
    "ISFP": "예술적 감각이 뛰어나고 조화로운 관계를 중시합니다. 현재에 충실하며 자신의 가치에 따라 행동합니다.",
    "ESFJ": "친절하고 협조적이며 타인의 필요에 민감합니다. 조화와 협력을 중시하고 구체적인 방법으로 도움을 줍니다.",
    "ESFP": "사교적이고 열정적이며 즐거움을 추구합니다. 현재에 충실하고 실용적이며 타인과의 교류를 즐깁니다."
}

MBTI_LEARNING_STYLES = {
    # 직관형(N) + 사고형(T) 조합
    "INTJ": "독립적인 학습을 선호하며, 개념적이고 이론적인 지식에 관심이 많습니다. 체계적인 계획을 세우고 목표 달성을 위해 집중합니다.",
    "INTP": "개념과 원리를 이해하는 것을 중요시하며, 스스로 문제를 해결하는 것을 좋아합니다. 새로운 아이디어와 가능성을 탐구하는 데 관심이 많습니다.",
    "ENTJ": "구조화된 학습 환경을 선호하며, 명확한 목표와 기준을 중시합니다. 논리적인 추론과 체계적인 접근을 통해 학습합니다.",
    "ENTP": "다양한 주제에 관심을 가지며, 토론과 논쟁을 통해 배우는 것을 좋아합니다. 창의적인 문제 해결과 새로운 접근법을 시도하는 것을 즐깁니다.",

    # 직관형(N) + 감정형(F) 조합
    "INFJ": "의미 있는 학습과 개인적 성장을 중요시합니다. 조용한 환경에서 깊이 있는 사고와 성찰을 통해 학습하는 것을 선호합니다.",
    "INFP": "자신의 가치와 관심사에 맞는 주제를 학습할 때 가장 동기부여가 됩니다. 창의적이고 독립적인 학습 방식을 선호합니다.",
    "ENFJ": "협력적인 학습 환경을 선호하며, 다른 사람들과 함께 배우고 성장하는 것을 중요시합니다. 긍정적인 피드백과 인정을 통해 동기부여를 받습니다.",
    "ENFP": "다양한 학습 방식과 창의적인 접근을 즐깁니다. 흥미로운 주제와 새로운 경험을 통해 배우는 것을 선호합니다.",

    # 감각형(S) + 사고형(T) 조합
    "ISTJ": "체계적이고 순차적인 학습을 선호하며, 명확한 지시와 구체적인 예시를 중요시합니다. 실용적인 지식과 기술 습득에 집중합니다.",
    "ISTP": "실습과 체험을 통한 학습을 선호하며, 실제 상황에서 문제를 해결하는 데 능숙합니다. 구체적이고 현실적인 학습 내용을 선호합니다.",
    "ESTJ": "구조화된 환경에서 명확한 목표와 일정에 따라 학습하는 것을 선호합니다. 실용적이고 즉시 적용 가능한 지식을 중요시합니다.",
    "ESTP": "활동적이고 실제적인 학습을 선호하며, 직접 경험하고 시도해보는 것을 통해 가장 잘 배웁니다. 현재 중심적이고 실용적인 접근을 선호합니다.",

    # 감각형(S) + 감정형(F) 조합
    "ISFJ": "체계적이고 구체적인 지시를 선호하며, 정보를 순차적으로 처리합니다. 안정적인 학습 환경과 실용적인 적용을 중요시합니다.",
    "ISFP": "자유로운 학습 환경에서 자신의 속도에 맞춰 배우는 것을 선호합니다. 예술적이고 실제적인 경험을 통한 학습에 흥미를 느낍니다.",
    "ESFJ": "협력적인 학습 환경과 그룹 활동을 통해 잘 배웁니다. 구체적인 피드백과 격려가 동기부여에 도움이 됩니다.",
    "ESFP": "활동적이고 사회적인 학습 환경을 선호하며, 게임이나 그룹 활동을 통한 학습에 흥미를 느낍니다. 즉각적인 결과와 피드백을 중요시합니다."
}

MBTI_COMMUNICATION_STYLES = {
    # 직관형(N) + 사고형(T) 조합
    "INTJ": "논리적이고 직접적인 의사소통을 선호합니다. 감정보다 사실과 분석에 중점을 두며, 효율적인 소통을 중시합니다.",
    "INTP": "개념적이고 분석적인 대화를 즐기며, 정확한 용어 사용과 논리적 일관성을 중요시합니다. 때로는 추상적인 아이디어에 몰두할 수 있습니다.",
    "ENTJ": "직설적이고 명확한 의사소통을 선호합니다. 목표 지향적인 대화와 효율적인 의사결정을 중시합니다.",
    "ENTP": "지적인 토론과 논쟁을 즐기며, 새로운 아이디어와 가능성을 탐구하는 대화를 선호합니다. 때로는 도전적인 질문을 통해 소통합니다.",

    # 직관형(N) + 감정형(F) 조합
    "INFJ": "깊이 있고 의미 있는 대화를 선호하며, 타인의 감정과 필요에 민감합니다. 조화로운 소통과 진정성을 중요시합니다.",
    "INFP": "가치와 신념에 기반한 소통을 중시하며, 타인의 감정과 관점을 존중합니다. 진정성 있는 개인적 표현을 선호합니다.",
    "ENFJ": "타인에게 영감을 주고 지원하는 의사소통 스타일을 가지며, 조화와 긍정적 관계 구축에 중점을 둡니다.",
    "ENFP": "열정적이고 표현력이 풍부한 소통을 선호하며, 다양한 관점과 가능성을 탐구하는 대화를 즐깁니다. 타인에게 영감을 주는 방식으로 소통합니다.",

    # 감각형(S) + 사고형(T) 조합
    "ISTJ": "사실적이고 구체적인 의사소통을 선호하며, 명확하고 직접적인 정보 전달을 중시합니다. 불필요한 세부사항 없이 요점을 전달합니다.",
    "ISTP": "간결하고 실용적인 소통을 선호하며, 문제 해결 중심의 대화에 집중합니다. 과도한 감정 표현보다 명확한 사실 전달을 중시합니다.",
    "ESTJ": "구조적이고 직접적인 의사소통을 선호하며, 명확한 지시와 기대치를 전달합니다. 효율성과 결과 중심의 소통을 중시합니다.",
    "ESTP": "직설적이고 현실적인 소통 스타일을 가지며, 실용적인 해결책과 즉각적인 행동을 중시합니다. 유머와 재치를 통해 소통하는 경우가 많습니다.",

    # 감각형(S) + 감정형(F) 조합
    "ISFJ": "따뜻하고 지지적인 의사소통을 선호하며, 타인의 필요와 감정에 세심한 주의를 기울입니다. 조화로운 관계를 중시합니다.",
    "ISFP": "부드럽고 비판단적인 소통 방식을 가지며, 타인의 자율성을 존중합니다. 갈등을 피하고 조화를 추구하는 경향이 있습니다.",
    "ESFJ": "친절하고 협조적인 의사소통을 선호하며, 타인의 감정과 필요에 민감하게 반응합니다. 구체적인 정보와 개인적 관심을 통해 소통합니다.",
    "ESFP": "활기차고 표현력이 풍부한 소통을 선호하며, 즐거운 대화와 실제 경험 공유를 통해 관계를 구축합니다. 개방적이고 수용적인 태도로 소통합니다."
}

PARENT_CONSULTATION_ATTITUDES = {
    "방어적 태도": "교사의 피드백이나 제안에 방어적으로 반응하며, 자녀에 대한 비판을 개인적인 공격으로 받아들이는 경향이 있습니다. 자녀의 문제를 인정하기 어려워하고 외부 요인을 탓하는 경우가 많습니다.",
    "협력적 태도": "교사와의 파트너십을 중요시하며, 자녀의 발전을 위해 적극적으로 협력하려는 의지를 보입니다. 개방적인 소통과 공동 문제 해결을 추구합니다.",
    "요구적 태도": "구체적인 해결책과 즉각적인 결과를 기대하며, 교사에게 명확한 행동 계획을 요구합니다. 자녀의 학업 성취나 문제 해결에 대한 기대치가 높습니다.",
    "수동적 태도": "상담 과정에서 주로 듣는 입장을 취하며, 교사의 지시나 조언을 수용하지만 적극적인 의견 개진이나 질문이 적습니다.",
    "회의적 태도": "교육 시스템이나 교사의 접근 방식에 의구심을 가지며, 제안된 해결책의 효과성에 대해 의문을 제기합니다. 증거나 구체적인 사례를 요구하는 경향이 있습니다.",
    "정서적 태도": "상담 중 감정 표현이 풍부하며, 자녀의 문제나 상황에 대해 정서적으로 반응합니다. 공감과 정서적 지지를 중요시합니다.",
    "분석적 태도": "문제의 원인과 해결책을 논리적으로 분석하려고 하며, 데이터나 객관적 정보에 기반한 접근을 선호합니다. 감정보다는 사실에 초점을 맞춥니다.",
    "회피적 태도": "민감한 주제나 어려운 대화를 피하려는 경향이 있으며, 자녀의 문제를 최소화하거나 다른 화제로 전환하려고 합니다."
}

EDUCATIONAL_THEORIES = {
    "인지발달이론": {
        "설명": "피아제의 인지발달이론에 따르면 아동은 감각운동기, 전조작기, 구체적 조작기, 형식적 조작기를 거치며 발달합니다. 각 단계에서 아동은 다른 방식으로 세상을 이해하고 탐색합니다.",
        "적용": "학생의 인지발달 단계에 맞는 교육 방법을 제공하는 것이 중요합니다. 추상적 개념은 형식적 조작기에 도달한 후에 더 잘 이해할 수 있습니다."
    },
    "사회문화이론": {
        "설명": "비고츠키의 사회문화이론은 학습이 사회적 상호작용과 문화적 맥락 안에서 이루어진다고 강조합니다. 근접발달영역(ZPD)과 비계설정(scaffolding) 개념은 학생의 잠재력을 이끌어내는 데 중요합니다.",
        "적용": "학생의 현재 발달 수준과 잠재적 발달 수준 사이에서 적절한 도전과 지원을 제공하는 것이 효과적입니다."
    },
    "다중지능이론": {
        "설명": "가드너의 다중지능이론에 따르면 인간은 언어적, 논리-수학적, 공간적, 신체-운동적, 음악적, 대인관계적, 자기성찰적, 자연친화적 지능을 가지고 있으며, 각 개인마다 강점과 약점이 다릅니다.",
        "적용": "학생의 다양한 지능 유형을 고려한 다양한 교수법과 평가 방법을 활용하는 것이 모든 학생의 성장을 촉진할 수 있습니다."
    },
    "학습유형이론": {
        "설명": "학습유형이론은 개인이 정보를 받아들이고 처리하는 방식이 다르다고 설명합니다. 주요 학습 유형으로는 시각적, 청각적, 읽기/쓰기, 신체적(VARK) 등이 있습니다.",
        "적용": "학생의 선호하는 학습 방식을 고려하여 다양한 교수 방법을 활용하면 학습 효과를 높일 수 있습니다."
    },
    "자기결정성이론": {
        "설명": "데시와 라이언의 자기결정성이론은 내재적 동기를 촉진하는 세 가지 기본 심리적 욕구(자율성, 유능감, 관계성)를 강조합니다.",
        "적용": "학생들에게 선택권을 제공하고, 성공 경험을 통해 유능감을 느끼게 하며, 지지적인 관계를 형성하는 것이 학습 동기를 높이는 데 중요합니다."
    },
    "성장마인드셋": {
        "설명": "드웩의 성장마인드셋 이론은 능력이 고정된 것이 아니라 노력과 학습을 통해 발전할 수 있다는 믿음의 중요성을 강조합니다.",
        "적용": "결과보다 과정과 노력을 칭찬하고, 실패를 학습 기회로 인식하도록 돕는 것이 중요합니다."
    },
    "긍정적 행동지원": {
        "설명": "긍정적 행동지원(PBS)은 문제 행동을 예방하고 긍정적 행동을 강화하는 체계적인 접근법입니다.",
        "적용": "명확한 기대치 설정, 긍정적 강화, 일관된 결과 제공을 통해 학생의 바람직한 행동을 증진시킬 수 있습니다."
    },
    "부모양육유형": {
        "설명": "바움린드의 부모양육유형 이론은 권위적(authoritative), 권위주의적(authoritarian), 허용적(permissive), 방임적(neglectful) 양육 스타일을 구분합니다.",
        "적용": "적절한 한계 설정과 높은 반응성을 결합한 권위적 양육 스타일이 아동의 발달에 가장 긍정적인 영향을 미치는 것으로 알려져 있습니다."
    },
    "애착이론": {
        "설명": "볼비와 에인스워스의 애착이론은 영유아와 주 양육자 간의 정서적 유대 관계가 아동의 사회적, 정서적 발달에 중요한 영향을 미친다고 설명합니다.",
        "적용": "안정적 애착은 아동의 자신감, 탐색 능력, 대인관계 형성에 긍정적인 영향을 줍니다. 학교에서도 안정적이고 지지적인 관계를 제공하는 것이 중요합니다."
    },
    "정서지능": {
        "설명": "골먼의 정서지능 이론은 자신과 타인의 감정을 인식하고 관리하는 능력의 중요성을 강조합니다.",
        "적용": "학생들이 자신의 감정을 인식하고 적절히 표현하며, 타인의 감정을 공감하고 대인관계를 효과적으로 관리하는 기술을 발달시키도록 돕는 것이 중요합니다."
    }
}

CONSULTATION_SCENARIO = """
1. 상담 시작 신뢰 형성
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

def get_max_tokens():
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    if conv_length == "단계 1 (짧음)":
        return 400
    elif conv_length == "단계 2 (보통)":
        return 800
    elif conv_length == "단계 3 (긴 대화)":
        return 1200
    else:
        return 800

def get_summary_max_tokens():
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    if conv_length == "단계 1 (짧음)":
        return 100
    elif conv_length == "단계 2 (보통)":
        return 150
    elif conv_length == "단계 3 (긴 대화)":
        return 200
    else:
        return 150


# ------------------------
# 새로 추가할 유틸 함수: Big Five 정보 요약
# ------------------------
def generate_big_five_summary(big_five_dict):
    """
    선택된 5개 Big Five 수준 정보를 받아, 
    각 범주와 설명을 요약한 문자열을 생성합니다.
    """
    summary_lines = []
    for trait_category, trait_level in big_five_dict.items():
        trait_desc = BIG_FIVE_TRAITS.get(trait_level, "정보 없음")
        summary_lines.append(f"- **{trait_category}**: {trait_level}\n  {trait_desc}")
    # 줄바꿈과 함께 요약 본문을 하나로 합침
    summary_text = "\n".join(summary_lines)
    return summary_text


# ------------------------
# 시스템 프롬프트 생성 로직
# ------------------------
def generate_system_prompt(data):
    """
    (수정됨) Big Five 전체 정보를 반영하여 프롬프트 생성
    """
    # 1) Big Five 요약문 생성
    big_five_dict = data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else "빅5 성격특성이 설정되지 않았습니다."

    # 2) 기존 구성
    prompt = f"""다음 상담 정보를 바탕으로 상담을 진행하세요.

[상담 정보]
- 학교급: {data.get('school_type', '')}
- 성별: {data.get('gender', '')}
- 학년: {data.get('grade', '')}
- Big Five 성격 특성:
{big_five_text}

- 주요 상담 내용: {data.get('counseling_issue', '')}
- 학부모 상담 태도: {data.get('parent_attitude', '')}
- 학부모 태도 특성: {PARENT_CONSULTATION_ATTITUDES.get(data.get('parent_attitude', ''), '상담 태도 정보가 없습니다.')}
"""

    # 3) MBTI 정보 추가
    student_mbti = data.get('student_mbti', '')
    teacher_mbti = data.get('teacher_mbti', '')

    if student_mbti:
        prompt += f"""
[학생 MBTI 정보]
- MBTI 유형: {student_mbti}
- 성격 특성: {MBTI_CHARACTERISTICS.get(student_mbti, '')}
- 학습 스타일: {MBTI_LEARNING_STYLES.get(student_mbti, '')}
- 의사소통 방식: {MBTI_COMMUNICATION_STYLES.get(student_mbti, '')}
"""

    if teacher_mbti:
        prompt += f"""
[교사 MBTI 정보]
- MBTI 유형: {teacher_mbti}
- 성격 특성: {MBTI_CHARACTERISTICS.get(teacher_mbti, '')}
- 의사소통 방식: {MBTI_COMMUNICATION_STYLES.get(teacher_mbti, '')}
"""

    return prompt


def summarize_chat_history(messages):
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    if conv_length == "단계 1 (짧음)":
        length_guideline = "매우 간결하게 핵심만 요약해주세요. 50-70단어 이내로 제한하세요."
    elif conv_length == "단계 2 (보통)":
        length_guideline = "적절한 길이로 중요한 내용을 포함해 요약해주세요. 100-150단어 정도가 적절합니다."
    elif conv_length == "단계 3 (긴 대화)":
        length_guideline = "중요한 세부 사항을 포함하여 포괄적으로 요약해주세요. 200-250단어까지 작성 가능합니다."
    else:
        length_guideline = "적절한 길이로 요약해주세요."

    conversation_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in messages]
    )
    summarization_prompt = (
        "다음 대화 내용을 요약해주세요. "
        "핵심 포인트와 중요한 정보를 포함하되, 불필요한 세부 사항은 생략하고 자연스럽게 연결되는 요약문을 작성해주세요.\n"
        f"요약 길이 지침: {length_guideline}\n\n"
        f"{conversation_text}"
    )
    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.5,
        max_tokens=get_summary_max_tokens(),
    )
    response = chat.invoke([{"role": "system", "content": summarization_prompt}])
    return response.content.strip()


def get_recent_context(chat_history):
    conversation_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    if conversation_length == "단계 1 (짧음)":
        max_messages = 4
    elif conversation_length == "단계 2 (보통)":
        max_messages = 6
    elif conversation_length == "단계 3 (긴 대화)":
        max_messages = 8
    else:
        max_messages = 6

    if len(chat_history) <= max_messages + 1:
        return chat_history
    else:
        system_message = chat_history[0]
        messages_to_summarize = chat_history[1:-max_messages]
        summary = summarize_chat_history(messages_to_summarize)
        summary_message = {"role": "system", "content": f"이전 대화 요약: {summary}"}
        return [system_message, summary_message] + chat_history[-max_messages:]


# ------------------------
# 역할별 system 프롬프트 생성 함수
# ------------------------
def generate_role_system_prompt(role, data):
    base_prompt = generate_system_prompt(data)

    # Big Five 전체 텍스트(추가 참조용)
    big_five_dict = data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    student_name = data.get("student_name", "")
    gender = data.get("gender", "")
    grade = data.get("grade", "")
    school_type = data.get("school_type", "")
    student_mbti = data.get("student_mbti", "")
    teacher_mbti = data.get("teacher_mbti", "")
    parent_attitude = data.get("parent_attitude", "")
    conv_length = data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, CONVERSATION_LENGTH_GUIDELINES["단계 2 (보통)"])

    if role == "선생님 -> 가상 학부모":
        role_prompt = (
            "당신은 인격과 개성이 뚜렷한 가상 학부모입니다. 선생님이 제공한 정보를 바탕으로, "
            f"{student_name}({gender}, {school_type} {grade})에 관한 대화를 진행합니다. "
            "자연스러운 일상대화 형식으로 자녀의 학교생활과 가정생활에 관한 고민이나 질문에 대해 진솔하게 답변하세요. "
            "딱딱한 '첫째, 둘째' 같은 나열식 표현 대신 대화체로 이야기하세요."
        )
        if parent_attitude:
            role_prompt += f"\n\n당신은 {parent_attitude} 성향의 학부모입니다. {PARENT_CONSULTATION_ATTITUDES.get(parent_attitude, '')} 이런 태도로 교사와의 상담에 임하세요."
        if student_mbti:
            role_prompt += f"\n\n자녀 {student_name}의 MBTI는 {student_mbti}입니다. 이 성격 특성을 고려한 고민과 질문을 자연스럽게 포함하세요."
        if big_five_dict:
            role_prompt += f"\n\n자녀가 보이는 Big Five 성격 특성:\n{big_five_text}\n위 특징들이 학교와 가정에서 어떻게 나타나는지 고민해보며 대화를 이어가세요."

        role_prompt += f"\n\n대화 길이 지침: {length_guideline}"

    elif role == "학부모 -> 가상 선생님":
        role_prompt = (
            "당신은 교육학과 아동발달 이론에 대한 전문성을 갖춘 선생님입니다. 학부모의 메시지에 응답할 때, "
            f"{student_name}({gender}, {school_type} {grade})에 관한 전문적 견해를 자연스럽게 드러내면서도 친근하고 공감적인 태도를 유지하세요. "
            "상담심리학, 발달심리학, 교육과정, 학습이론 등의 전문 지식을 활용하되, 이론명이나 학자 이름을 직접 언급하지 말고 실용적인 설명과 조언에 집중하세요. "
            "자연스러운 대화체로 학부모와 소통하며, 딱딱한 나열식 표현 대신 일상 대화처럼 대응하세요."
        )
        if parent_attitude:
            role_prompt += f"\n\n학부모님은 {parent_attitude} 성향을 보이고 있습니다. {PARENT_CONSULTATION_ATTITUDES.get(parent_attitude, '')} 이러한 태도에 맞춰 적절한 소통 전략을 사용하세요."
        if teacher_mbti:
            role_prompt += f"\n\n선생님으로서 당신의 MBTI는 {teacher_mbti}입니다. 이를 고려한 의사소통 방식을 자연스럽게 반영하세요."
        if student_mbti:
            role_prompt += f"\n\n{student_name} 학생의 MBTI는 {student_mbti}입니다. 이 성격 유형의 특성과 학습 스타일을 고려하여 학부모에게 조언하세요."
        if big_five_dict:
            role_prompt += f"\n\n학생이 보이는 Big Five 성격 특성:\n{big_five_text}\n이 특징들에 맞는 교육적 접근법을 학부모에게 안내하세요."

        role_prompt += f"\n\n대화 길이 지침: {length_guideline}"

    elif role == "학생 -> 가상 선생님":
        role_prompt = (
            "당신은 교육 이론과 학생 발달에 대한 깊은 이해를 갖춘 전문 교사입니다. 학생의 메시지에 응답할 때, "
            f"{school_type} {grade} {gender} 학생의 발달 수준에 맞게 교육학적 지식을 쉽고 친근하게 전달하세요. "
            "학생의 인지적, 정서적, 사회적 발달 단계를 고려하여, 이해하기 쉬운 언어로 정보를 제공하고, "
            "자기주도적 문제해결과 성장을 촉진하는 조언을 자연스러운 대화체로 제공하세요. "
            "딱딱한 나열식 표현 대신 친근한 대화체로 소통하세요."
        )
        if teacher_mbti:
            role_prompt += f"\n\n선생님으로서 당신의 MBTI는 {teacher_mbti}입니다. 이러한 성격 특성이 교육 방식과 학생 지도에 자연스럽게 반영되도록 하세요."
        if student_mbti:
            role_prompt += f"\n\n학생의 MBTI는 {student_mbti}입니다. 이 특성을 고려한 맞춤형 학습 전략과 의사소통 방식을 활용하세요."
        if big_five_dict:
            role_prompt += f"\n\n학생이 보이는 Big Five 성격 특성:\n{big_five_text}\n해당 성격적 특징에 맞춰 조언을 제공하세요."

        role_prompt += f"\n\n대화 길이 지침: {length_guideline}"

    elif role == "선생님 -> 가상 학생":
        role_prompt = (
            f"당신은 {school_type} {grade} {gender} 학생입니다. "
            "선생님이 제공한 정보를 참고하여, 자연스러운 대화체로 솔직하고 진솔한 답변을 작성하세요. "
            "딱딱한 나열식 표현 대신 또래 학생들이 실제로 사용할 법한 자연스러운 어투를 사용하세요."
        )
        if student_mbti:
            role_prompt += f"\n\n당신의 MBTI는 {student_mbti}입니다. 이 성격 유형의 특성을 대화에 반영하세요."
        if big_five_dict:
            role_prompt += f"\n\n당신이 보이는 Big Five 성격 특성:\n{big_five_text}\n이 특징에 맞춰 학교생활과 대화 방식을 표현하세요."

        role_prompt += f"\n\n대화 길이 지침: {length_guideline}"

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


# ------------------------
# 대화 마무리/종료 메시지 생성 함수
# ------------------------
def generate_closing_message(role, chat_history):
    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")
    student_mbti = st.session_state.data.get("student_mbti", "")
    teacher_mbti = st.session_state.data.get("teacher_mbti", "")
    parent_attitude = st.session_state.data.get("parent_attitude", "")
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, CONVERSATION_LENGTH_GUIDELINES["단계 2 (보통)"])

    # Big Five 전체 텍스트
    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    closing_instruction = (
        "대화를 마무리하는 말로, 오늘 상담에 참여해 주셔서 감사합니다. "
        "앞으로도 지속적으로 소통하며 도와드리겠습니다. 좋은 하루 보내세요."
        f"\n\n대화 길이 지침: {length_guideline}"
    )

    if role == "선생님 -> 가상 학부모":
        closing_instruction = (
            f"{student_name}의 학부모로서 선생님과의 상담을 마무리하는 인사말을 자연스러운 대화체로 작성해주세요. "
            "감사 인사와 앞으로의 소통에 대한 기대를 포함하면 좋겠습니다."
        )
        if parent_attitude:
            closing_instruction += f"\n\n{parent_attitude} 성향의 학부모로서, 이러한 태도가 마무리 인사에도 자연스럽게 드러나게 해주세요."
        if student_mbti:
            closing_instruction += f"\n\n자녀의 MBTI({student_mbti}) 특성을 고려한 언급을 자연스럽게 포함해주세요."
        if big_five_dict:
            closing_instruction += f"\n\n자녀가 보이는 Big Five 성격 특성:\n{big_five_text}\n이 부분을 인식하고 있다는 뉘앙스를 자연스럽게 드러내주세요."
        closing_instruction += f"\n\n대화 길이 지침: {length_guideline}"

    elif role == "학부모 -> 가상 선생님":
        closing_instruction = (
            f"교육 전문가로서 {student_name}의 학부모와의 상담을 마무리하는 인사말을 자연스러운 대화체로 작성해주세요. "
            "학부모와의 지속적인 소통과 협력이 학생의 성장에 중요함을 언급하고, "
            "앞으로도 도움을 드릴 수 있다는 의지를 표현하면 좋겠습니다."
        )
        if parent_attitude:
            closing_instruction += f"\n\n학부모님은 {parent_attitude} 성향을 보이고 있습니다. 이에 맞춰 배려 깊은 마무리 인사를 해주세요."
        if teacher_mbti:
            closing_instruction += f"\n\n선생님의 MBTI({teacher_mbti}) 특성이 은은하게 드러나는 표현을 포함해주세요."
        if student_mbti:
            closing_instruction += f"\n\n학생의 MBTI({student_mbti}) 특성을 고려한 조언을 자연스럽게 포함해주세요."
        if big_five_dict:
            closing_instruction += f"\n\n학생이 보이는 Big Five 성격 특성:\n{big_five_text}\n해당 특성에 대한 교육적 조언을 마무리 멘트에 녹여주세요."
        closing_instruction += f"\n\n대화 길이 지침: {length_guideline}"

    elif role == "학생 -> 가상 선생님":
        closing_instruction = (
            f"{school_type} {grade} 학생과의 상담을 마무리하는 교사로서의 인사말을 자연스러운 대화체로 작성해주세요. "
            "학생의 성장을 지원하고 앞으로도 도움이 필요하면 언제든 찾아오라는 메시지를 포함하면 좋겠습니다."
        )
        if teacher_mbti:
            closing_instruction += f"\n\n선생님의 MBTI({teacher_mbti}) 특성이 은은하게 드러나는 표현을 포함해주세요."
        if student_mbti:
            closing_instruction += f"\n\n학생의 MBTI({student_mbti}) 특성을 고려한 조언이나 언급을 자연스럽게 포함해주세요."
        if big_five_dict:
            closing_instruction += f"\n\n학생이 보이는 Big Five 성격 특성:\n{big_five_text}\n이에 맞는 격려나 조언을 포함해주세요."
        closing_instruction += f"\n\n대화 길이 지침: {length_guideline}"

    elif role == "선생님 -> 가상 학생":
        closing_instruction = (
            f"{school_type} {grade} 학생으로서 선생님과의 상담을 마무리하는 인사말을 자연스러운 대화체로 작성해주세요. "
            "상담에 대한 간단한 소감과 감사 인사를 포함하면 좋겠습니다."
        )
        if student_mbti:
            closing_instruction += f"\n\n학생의 MBTI({student_mbti}) 특성을 자연스럽게 드러내는 표현과 어투를 사용해주세요."
        if big_five_dict:
            closing_instruction += f"\n\n본인의 Big Five 성격 특성:\n{big_five_text}\n이 드러나는 멘트를 포함해주세요."
        closing_instruction += f"\n\n대화 길이 지침: {length_guideline}"

    closing_prompt = f"이전 대화 내용을 참고하여, 자연스럽게 마무리하는 인사말을 작성해줘. 다음 문장을 참고해:\n{closing_instruction}"
    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"], 
        model="gpt-4o", 
        temperature=0.5, 
        max_tokens=get_summary_max_tokens()
    )
    response = chat.invoke([{"role": "system", "content": closing_prompt}])
    return response.content.strip()


# ------------------------
# 역할별 대화 생성 함수
# ------------------------
def generate_parent_response(chat_history):
    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")
    parent_attitude = st.session_state.data.get("parent_attitude", "미지정")
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, "")

    # Big Five 텍스트
    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    greeting_line = ""
    if not st.session_state.get("greeting_sent", False):
        greeting_line = "안녕하세요, 만나 뵙게 되어 반갑습니다. "
        st.session_state.greeting_sent = True

    parent_instruction = (
        greeting_line
        + f"[학부모 상담 태도: {parent_attitude}] "
        f"당신은 {student_name}({gender}, {school_type} {grade})의 학부모입니다. "
        f"{parent_attitude} 태도에 알맞은 스타일로 자연스럽게 대화하세요. "
        f"{PARENT_CONSULTATION_ATTITUDES.get(parent_attitude, '')} "
        "자녀의 학교생활이나 가정생활에 관한 구체적인 고민, 질문, 의견을 진솔하게 작성하세요. "
        "딱딱한 나열식 표현보다는 일상 대화체로 응답하세요. "
        f"\n\n대화 길이 지침: {length_guideline}\n"
        "한 번의 메시지에는 하나의 질문과 하나의 내용에 대해서만 이야기해 주세요."
    )

    if big_five_dict:
        parent_instruction += f"\n\n자녀가 보이는 Big Five 성격 특성:\n{big_five_text}\n이 성격적 특징들과 관련된 고민이나 질문을 자연스럽게 포함하세요."

    history = chat_history + [{"role": "system", "content": parent_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.6,
        max_tokens=get_max_tokens()
    )
    response = chat.invoke(recent_history)
    return response.content.strip()


def generate_teacher_response(chat_history):
    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")
    parent_attitude = st.session_state.data.get("parent_attitude", "")
    teacher_mbti = st.session_state.data.get("teacher_mbti", "")
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, "")

    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    greeting_line = ""
    if not st.session_state.get("greeting_sent", False):
        greeting_line = "안녕하세요, 만나 뵙게 되어 반갑습니다. "
        st.session_state.greeting_sent = True

    teacher_instruction = (
        greeting_line
        + f"당신은 {student_name}({gender}, {school_type} {grade})의 담임 선생님으로, 교육 경험과 전문성이 풍부합니다. "
        "학부모의 질문에 응답할 때, 교사로서의 전문적 견해와 관점을 자연스럽게 드러내되, "
        "학자의 이름이나 이론의 명칭을 직접 언급하지 말고 관찰한 내용과 실질적인 조언에 집중하세요. "
        "학부모가 제기한 질문에 직접적으로 답변하면서도, 전문 용어는 이해하기 쉽게 풀어서 설명하세요. "
        "친근하면서도 전문가로서의 신뢰감을 줄 수 있는 어조를 유지하시고, "
        "딱딱한 나열식 표현 대신 자연스러운 대화체로 소통해주세요.\n"
        f"\n대화 길이 지침: {length_guideline}\n"
        "한 번의 메시지에는 하나의 주제에 대해서만 답변해 주세요."
    )

    if parent_attitude:
        teacher_instruction += f"\n\n학부모님은 {parent_attitude} 태도를 보이고 있습니다. {PARENT_CONSULTATION_ATTITUDES.get(parent_attitude, '')} 이 점을 고려하여 대응하세요."

    if teacher_mbti:
        teacher_instruction += f"\n\n당신의 MBTI는 {teacher_mbti}입니다. 이 성격 유형의 소통 방식을 학부모 상담에 적절히 반영하세요."

    if big_five_dict:
        teacher_instruction += f"\n\n학생이 보이는 Big Five 성격 특성:\n{big_five_text}\n여기에 맞는 교육적 접근법과 전략을 제시해주세요."

    history = chat_history + [{"role": "system", "content": teacher_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=get_max_tokens()
    )
    response = chat.invoke(recent_history)
    return response.content.strip()


def generate_teacher_response_for_student(chat_history):
    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")
    student_mbti = st.session_state.data.get("student_mbti", "")
    teacher_mbti = st.session_state.data.get("teacher_mbti", "")
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, "")

    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    greeting_line = ""
    if not st.session_state.get("greeting_sent", False):
        greeting_line = "안녕하세요, 만나서 반갑습니다. "
        st.session_state.greeting_sent = True

    teacher_instruction = (
        greeting_line
        + f"당신은 {school_type} {grade} {gender} 학생을 위한 선생님입니다. "
        "학생의 메시지에 응답할 때, 교육자로서의 전문적 견해와 관점을 학생 수준에 맞게 전달하세요. "
        "학자의 이름이나 이론 명칭을 직접 언급하지 말고, 실질적인 조언과 구체적인 사례를 제시해주세요. "
        "학생의 인지적, 정서적 발달 단계를 고려하여 친근하고 긍정적인 대화체로 안내해주세요.\n"
        f"\n대화 길이 지침: {length_guideline}\n"
        "한 번의 메시지에는 하나의 주제에 대해서만 답변해 주세요."
    )

    if teacher_mbti:
        teacher_instruction += f"\n\n당신의 MBTI는 {teacher_mbti}입니다. 이 점을 고려하여 의사소통 방식을 구성하세요."

    if student_mbti:
        teacher_instruction += f"\n\n학생의 MBTI는 {student_mbti}입니다. 이 성격 특성에 맞는 학습 조언과 의사소통 방식을 사용하세요."

    if big_five_dict:
        teacher_instruction += f"\n\n학생이 보이는 Big Five 성격 특성:\n{big_five_text}\n이에 맞는 맞춤형 도움을 제공하세요."

    history = chat_history + [{"role": "system", "content": teacher_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=get_max_tokens()
    )
    response = chat.invoke(recent_history)
    return response.content.strip()


def generate_student_response(chat_history):
    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")
    student_mbti = st.session_state.data.get("student_mbti", "")
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, "")

    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    greeting_line = ""
    if not st.session_state.get("greeting_sent", False):
        greeting_line = "안녕하세요, 만나 뵙게 되어 반갑습니다. "
        st.session_state.greeting_sent = True

    student_instruction = (
        greeting_line
        + f"당신은 {school_type} {grade} {gender} 학생입니다. "
        "선생님의 메시지에 대해 자연스럽게 이어지는 질문이나 의견을 진솔하게 표현하세요. "
        "실제 학생이 말할 법한 일상 대화체를 사용하고, 딱딱한 나열식 표현을 피해주세요.\n"
        f"\n대화 길이 지침: {length_guideline}\n"
        "한 번의 메시지에는 하나의 질문과 하나의 주제에 대해서만 이야기해 주세요."
    )

    if student_mbti:
        student_instruction += f"\n\n당신의 MBTI는 {student_mbti}입니다. 이 성격 유형의 특성이 자연스럽게 드러나도록 응답하세요."

    if big_five_dict:
        student_instruction += f"\n\n당신이 보이는 Big Five 성격 특성:\n{big_five_text}\n이 부분도 대화에서 자연스럽게 표현하세요."

    history = chat_history + [{"role": "system", "content": student_instruction}]
    recent_history = get_recent_context(history)
    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.6,
        max_tokens=get_max_tokens()
    )
    response = chat.invoke(recent_history)
    return response.content.strip()


# ------------------------
# 추천 답변(Teacher->학부모, Teacher->학생 등) 함수
# ------------------------
def generate_teacher_input_suggestions(chat_history):
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, "")

    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")
    parent_attitude = st.session_state.data.get("parent_attitude", "")
    teacher_mbti = st.session_state.data.get("teacher_mbti", "")

    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    suggestion_instruction = (
        "당신은 교육 전문가입니다. 지금까지의 대화 내용을 바탕으로, "
        f"{student_name}({gender}, {school_type} {grade})에 관한 학부모 상담 대화에서 활용할 수 있는 답변 예시 3가지를 생성해주세요. "
        "각 답변에는 교육학, 교육심리학, 상담심리학, 뇌과학 이론적 배경이나 근거를 포함하세요. "
        "응답은 자연스러운 일상대화 형식으로, 교육 전문성이 간접적으로 드러나게 작성하세요. "
        "딱딱한 '첫째, 둘째' 같은 나열식 표현은 사용하지 마세요. "
        "학자의 이름이나 이론 명칭은 직접 언급하지 말고, 실용적인 관찰과 조언에 집중하세요. "
        f"\n\n대화 길이 지침: {length_guideline}\n"
    )
    if parent_attitude:
        suggestion_instruction += f"\n학부모님은 {parent_attitude} 태도를 보입니다. {PARENT_CONSULTATION_ATTITUDES.get(parent_attitude, '')}"
    if teacher_mbti:
        suggestion_instruction += f"\n당신은 {teacher_mbti} 성향의 교사입니다. 이 성격 유형을 상담에 반영하세요."
    if big_five_dict:
        suggestion_instruction += f"\n학생이 보이는 Big Five 성격 특성:\n{big_five_text}\n이를 고려한 교육적 조언을 제시하세요."

    suggestion_instruction += (
        "\n상담 정보는 포함하지 말고, 추천 답변은 다음의 형식을 따라주세요:\n\n"
        "【추천 답변 A】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【교육학적 근거 A】\n"
        "[이 답변에 적용된 교육학, 상담심리학, 발달심리학, 뇌과학적 근거 이론이나 접근법에 대한 간략한 설명]\n\n"
        "【추천 답변 B】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【교육학적 근거 B】\n"
        "[이 답변에 적용된 교육학, 상담심리학, 발달심리학, 뇌과학적 근거 이론이나 접근법에 대한 간략한 설명]\n\n"
        "【추천 답변 C】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【교육학적 근거 C】\n"
        "[이 답변에 적용된 교육학, 상담심리학, 발달심리학, 뇌과학적 근거 이론이나 접근법에 대한 간략한 설명]"
    )

    history = chat_history + [{"role": "system", "content": suggestion_instruction}]
    recent_history = get_recent_context(history)

    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=get_max_tokens()
    )
    response = chat.invoke(recent_history)
    suggestions_text = response.content.strip()

    suggestions = []
    if "【추천 답변 A】" in suggestions_text:
        parts_a = suggestions_text.split("【추천 답변 A】")
        if len(parts_a) > 1:
            parts_a = parts_a[1].split("【추천 답변 B】")
            suggestion_a = parts_a[0].strip()
            if "【교육학적 근거 A】" in suggestion_a:
                answer_a, theory_a = suggestion_a.split("【교육학적 근거 A】")
                suggestions.append({
                    "answer": "【추천 답변 A】\n" + answer_a.strip(),
                    "theory": "【교육학적 근거】\n" + theory_a.strip()
                })

    if "【추천 답변 B】" in suggestions_text:
        parts_b = suggestions_text.split("【추천 답변 B】")
        if len(parts_b) > 1:
            parts_b = parts_b[1].split("【추천 답변 C】") if "【추천 답변 C】" in parts_b[1] else [parts_b[1]]
            suggestion_b = parts_b[0].strip()
            if "【교육학적 근거 B】" in suggestion_b:
                answer_b, theory_b = suggestion_b.split("【교육학적 근거 B】")
                suggestions.append({
                    "answer": "【추천 답변 B】\n" + answer_b.strip(),
                    "theory": "【교육학적 근거】\n" + theory_b.strip()
                })

    if "【추천 답변 C】" in suggestions_text:
        parts_c = suggestions_text.split("【추천 답변 C】")
        if len(parts_c) > 1:
            suggestion_c = parts_c[1].strip()
            if "【교육학적 근거 C】" in suggestion_c:
                answer_c, theory_c = suggestion_c.split("【교육학적 근거 C】")
                suggestions.append({
                    "answer": "【추천 답변 C】\n" + answer_c.strip(),
                    "theory": "【교육학적 근거】\n" + theory_c.strip()
                })

    if not suggestions:
        parts = suggestions_text.split("\n\n")
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                suggestions.append({
                    "answer": parts[i].strip(),
                    "theory": parts[i+1].strip()
                })
    return suggestions


def generate_teacher_input_suggestions_for_student(chat_history):
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, "")

    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")
    teacher_mbti = st.session_state.data.get("teacher_mbti", "")

    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    suggestion_instruction = (
        "당신은 교육 전문가입니다. 지금까지의 대화 내용을 바탕으로, "
        f"{school_type} {grade} {gender}와의 상담 대화에서 활용할 수 있는 선생님 답변 예시 3가지를 생성해주세요. "
        "각 답변에는 교육학, 교육심리학, 뇌과학적 이론적 배경이나 근거를 포함하세요. "
        "응답은 자연스러운 일상대화 형식으로, 교육 전문성이 간접적으로 드러나게 작성하세요. "
        "딱딱한 나열식 표현은 사용하지 마세요. "
        "학생들이 실제로 편안하게 대화할 수 있는 친근한 어투로 작성하세요. "
        "학자의 이름이나 이론 명칭은 직접 언급하지 말고, 실용적인 관찰과 조언에 집중하세요. "
        f"\n\n대화 길이 지침: {length_guideline}\n"
    )
    if teacher_mbti:
        suggestion_instruction += f"\n당신은 {teacher_mbti} 성향의 교사입니다. 이 성격 유형의 특성을 상담에 반영하세요."
    if big_five_dict:
        suggestion_instruction += f"\n학생이 보이는 Big Five 성격 특성:\n{big_five_text}\n이를 고려한 교육적 조언을 학생의 눈높이에 맞춰 제시하세요."

    suggestion_instruction += (
        "\n상담 정보는 포함하지 말고, 추천 답변은 다음의 형식을 따라주세요:\n\n"
        "【추천 답변 A】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【교육학적 근거 A】\n"
        "[이 답변에 적용된 교육학, 상담심리학, 발달심리학, 뇌과학적 근거 이론이나 접근법에 대한 간략한 설명]\n\n"
        "【추천 답변 B】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【교육학적 근거 B】\n"
        "[이 답변에 적용된 교육학, 상담심리학, 발달심리학, 뇌과학적 근거 이론이나 접근법에 대한 간략한 설명]\n\n"
        "【추천 답변 C】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【교육학적 근거 C】\n"
        "[이 답변에 적용된  교육학, 상담심리학, 발달심리학, 뇌과학적 근거 이론이나 접근법에 대한 간략한 설명]"
    )

    history = chat_history + [{"role": "system", "content": suggestion_instruction}]
    recent_history = get_recent_context(history)

    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=get_max_tokens()
    )
    response = chat.invoke(recent_history)
    suggestions_text = response.content.strip()

    suggestions = []
    if "【추천 답변 A】" in suggestions_text:
        parts_a = suggestions_text.split("【추천 답변 A】")
        if len(parts_a) > 1:
            parts_a = parts_a[1].split("【추천 답변 B】")
            suggestion_a = parts_a[0].strip()
            if "【교육학적 근거 A】" in suggestion_a:
                answer_a, theory_a = suggestion_a.split("【교육학적 근거 A】")
                suggestions.append({
                    "answer": "【추천 답변 A】\n" + answer_a.strip(),
                    "theory": "【교육학적 근거】\n" + theory_a.strip()
                })

    if "【추천 답변 B】" in suggestions_text:
        parts_b = suggestions_text.split("【추천 답변 B】")
        if len(parts_b) > 1:
            parts_b = parts_b[1].split("【추천 답변 C】") if "【추천 답변 C】" in parts_b[1] else [parts_b[1]]
            suggestion_b = parts_b[0].strip()
            if "【교육학적 근거 B】" in suggestion_b:
                answer_b, theory_b = suggestion_b.split("【교육학적 근거 B】")
                suggestions.append({
                    "answer": "【추천 답변 B】\n" + answer_b.strip(),
                    "theory": "【교육학적 근거】\n" + theory_b.strip()
                })

    if "【추천 답변 C】" in suggestions_text:
        parts_c = suggestions_text.split("【추천 답변 C】")
        if len(parts_c) > 1:
            suggestion_c = parts_c[1].strip()
            if "【교육학적 근거 C】" in suggestion_c:
                answer_c, theory_c = suggestion_c.split("【교육학적 근거 C】")
                suggestions.append({
                    "answer": "【추천 답변 C】\n" + answer_c.strip(),
                    "theory": "【교육학적 근거】\n" + theory_c.strip()
                })

    if not suggestions:
        parts = suggestions_text.split("\n\n")
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                suggestions.append({
                    "answer": parts[i].strip(),
                    "theory": parts[i+1].strip()
                })
    return suggestions


def generate_parent_input_suggestions(chat_history):
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, "")

    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")
    parent_attitude = st.session_state.data.get("parent_attitude", "미지정")

    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    suggestion_instruction = (
        "당신은 가상의 학부모입니다. 지금까지의 대화 내용을 반영하여, "
        f"{parent_attitude} 태도의 {student_name}({gender}, {school_type} {grade})의 학부모로서 "
        "선생님과의 상담 대화에서 말할 수 있는 추천 대화 예시 3가지를 생성해주세요. "
        "실제 학부모가 편안하게 대화할 수 있는 친근한 어투로 작성하세요. "
        f"\n\n대화 길이 지침: {length_guideline}\n"
    )
    if parent_attitude:
        suggestion_instruction += f"\n{parent_attitude} 상담 태도: {PARENT_CONSULTATION_ATTITUDES.get(parent_attitude, '')}"
    if big_five_dict:
        suggestion_instruction += f"\n자녀가 보이는 Big Five 성격 특성:\n{big_five_text}\n이와 관련된 고민이나 질문을 자연스럽게 녹여주세요."

    suggestion_instruction += (
        "\n상담 정보는 포함하지 말고, 추천 답변은 다음의 형식을 따라주세요:\n\n"
        "【추천 답변 A】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【추천 답변 B】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【추천 답변 C】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]"
    )

    history = chat_history + [{"role": "system", "content": suggestion_instruction}]
    recent_history = get_recent_context(history)

    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=get_max_tokens()
    )

    response = chat.invoke(recent_history)
    suggestions_text = response.content.strip()

    suggestions = []
    if "【추천 답변 A】" in suggestions_text:
        parts_a = suggestions_text.split("【추천 답변 A】")
        if len(parts_a) > 1:
            parts_a = parts_a[1].split("【추천 답변 B】") if "【추천 답변 B】" in parts_a[1] else [parts_a[1]]
            suggestion_a = parts_a[0].strip()
            suggestions.append({
                "answer": "【추천 답변 A】\n" + suggestion_a.strip(),
                "theory": ""
            })

    if "【추천 답변 B】" in suggestions_text:
        parts_b = suggestions_text.split("【추천 답변 B】")
        if len(parts_b) > 1:
            parts_b = parts_b[1].split("【추천 답변 C】") if "【추천 답변 C】" in parts_b[1] else [parts_b[1]]
            suggestion_b = parts_b[0].strip()
            suggestions.append({
                "answer": "【추천 답변 B】\n" + suggestion_b.strip(),
                "theory": ""
            })

    if "【추천 답변 C】" in suggestions_text:
        parts_c = suggestions_text.split("【추천 답변 C】")
        if len(parts_c) > 1:
            suggestion_c = parts_c[1].strip()
            suggestions.append({
                "answer": "【추천 답변 C】\n" + suggestion_c.strip(),
                "theory": ""
            })

    if not suggestions:
        parts = suggestions_text.split("\n\n")
        for part in parts:
            if part.strip():
                suggestions.append({
                    "answer": part.strip(),
                    "theory": ""
                })

    return suggestions


def generate_student_input_suggestions(chat_history):
    conv_length = st.session_state.data.get("conversation_length", "단계 2 (보통)")
    length_guideline = CONVERSATION_LENGTH_GUIDELINES.get(conv_length, "")

    student_name = st.session_state.data.get("student_name", "")
    gender = st.session_state.data.get("gender", "")
    grade = st.session_state.data.get("grade", "")
    school_type = st.session_state.data.get("school_type", "")

    big_five_dict = st.session_state.data.get("big_five_traits", {})
    big_five_text = generate_big_five_summary(big_five_dict) if big_five_dict else ""

    suggestion_instruction = (
        "당신은 가상의 학생입니다. 지금까지의 대화 내용을 반영하여, "
        f"{school_type} {grade} {gender}으로서 선생님과의 상담 대화에서 말할 수 있는 추천 대화 예시 3가지를 생성해주세요. "
        "딱딱한 나열식 표현은 사용하지 말고, 실제 학생이 말할 법한 자연스러운 어투로 작성하세요. "
        f"\n\n대화 길이 지침: {length_guideline}\n"
    )
    if big_five_dict:
        suggestion_instruction += f"\n당신이 보이는 Big Five 성격 특성:\n{big_five_text}\n이 성격적 특징을 반영한 질문이나 의견을 포함하세요."

    suggestion_instruction += (
        "\n상담 정보는 포함하지 말고, 추천 답변은 다음의 형식을 따라주세요:\n\n"
        "【추천 답변 A】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【추천 답변 B】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]\n\n"
        "【추천 답변 C】\n"
        "[여기에 추천 응답 내용을 자연스러운 대화체로 작성]"
    )

    history = chat_history + [{"role": "system", "content": suggestion_instruction}]
    recent_history = get_recent_context(history)

    chat = ChatOpenAI(
        openai_api_key=st.secrets["openai"]["api_key"],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=get_max_tokens()
    )

    response = chat.invoke(recent_history)
    suggestions_text = response.content.strip()

    suggestions = []
    if "【추천 답변 A】" in suggestions_text:
        parts_a = suggestions_text.split("【추천 답변 A】")
        if len(parts_a) > 1:
            parts_a = parts_a[1].split("【추천 답변 B】") if "【추천 답변 B】" in parts_a[1] else [parts_a[1]]
            suggestion_a = parts_a[0].strip()
            suggestions.append({
                "answer": "【추천 답변 A】\n" + suggestion_a.strip(),
                "theory": ""
            })

    if "【추천 답변 B】" in suggestions_text:
        parts_b = suggestions_text.split("【추천 답변 B】")
        if len(parts_b) > 1:
            parts_b = parts_b[1].split("【추천 답변 C】") if "【추천 답변 C】" in parts_b[1] else [parts_b[1]]
            suggestion_b = parts_b[0].strip()
            suggestions.append({
                "answer": "【추천 답변 B】\n" + suggestion_b.strip(),
                "theory": ""
            })

    if "【추천 답변 C】" in suggestions_text:
        parts_c = suggestions_text.split("【추천 답변 C】")
        if len(parts_c) > 1:
            suggestion_c = parts_c[1].strip()
            suggestions.append({
                "answer": "【추천 답변 C】\n" + suggestion_c.strip(),
                "theory": ""
            })

    if not suggestions:
        parts = suggestions_text.split("\n\n")
        for part in parts:
            if part.strip():
                suggestions.append({
                    "answer": part.strip(),
                    "theory": ""
                })

    return suggestions


def display_suggestions(suggestions):
    if not suggestions:
        st.info("추천 답변이 없습니다.")
        return

    st.markdown("### 💬 추천 대화 예시")

    # 현재 선택된 역할(예: "선생님 -> 가상 학부모" 등)
    role_mode = st.session_state.get("selected_role", "선생님 -> 가상 학부모")

    for i, suggestion in enumerate(suggestions):
        with st.container():
            # 추천 답변 본문 출력
            st.markdown(f"""
            <div class='suggestion-box'>
                {suggestion['answer']}
            </div>
            """, unsafe_allow_html=True)

            # 교육학적 근거가 있다면 expander에 출력
            if suggestion.get('theory'):
                with st.expander("🧠 교육학적 근거 보기"):
                    st.markdown(f"""
                    <div class='theory-box'>
                        <div class='theory-title'>교육학적 배경</div>
                        {suggestion['theory']}
                    </div>
                    """, unsafe_allow_html=True)

            # "이 답변 사용하기" 버튼
            if st.button(f"이 답변 사용하기", key=f"use_suggestion_{i}"):
                # (1) 불필요한 라벨 제거
                import re
                cleaned_answer = re.sub(r'【추천 답변 [ABC]】', '', suggestion['answer'])
                cleaned_answer = cleaned_answer.strip()

                # (2) 위에서 정규식으로 라벨을 제거한 텍스트만 user 메시지로 추가
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": cleaned_answer,
                    "mode": role_mode
                })

                # (3) 역할에 맞춰 모델 답변 생성
                if role_mode == "선생님 -> 가상 학부모":
                    with st.spinner("가상 학부모 응답 생성 중..."):
                        reply = generate_parent_response(st.session_state.chat_history)
                elif role_mode == "학부모 -> 가상 선생님":
                    with st.spinner("가상 선생님 응답 생성 중..."):
                        reply = generate_teacher_response(st.session_state.chat_history)
                elif role_mode == "학생 -> 가상 선생님":
                    with st.spinner("가상 선생님 응답 생성 중..."):
                        reply = generate_teacher_response_for_student(st.session_state.chat_history)
                else:  # "선생님 -> 가상 학생"
                    with st.spinner("가상 학생 응답 생성 중..."):
                        reply = generate_student_response(st.session_state.chat_history)

                # (4) 생성된 모델의 답변을 assistant 메시지로 추가
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": reply,
                    "mode": role_mode
                })

                # (5) 추천 답변 목록 닫기
                if "teacher_suggestions" in st.session_state:
                    del st.session_state["teacher_suggestions"]

                # (6) 화면 갱신
                st.rerun()

            # 추천 답변들을 구분하는 가로줄
            if i < len(suggestions) - 1:
                st.markdown("<hr style='margin: 15px 0; opacity: 0.2;'>", unsafe_allow_html=True)

def main():
    set_page_config()

    if "data" not in st.session_state:
        st.session_state.data = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    st.sidebar.markdown("<div class='sidebar-title'>👨‍👩‍👧‍👦 상담 정보 설정</div>", unsafe_allow_html=True)

    with st.sidebar.form("info_form"):
        st.markdown("#### 기본 정보")
        col1, col2 = st.columns(2)
        with col1:
            school_type = st.selectbox("학교급", ["초등학교", "중학교"])
            gender = st.selectbox("성별", ["남학생", "여학생"])
        with col2:
            if school_type == "초등학교":
                grade_options = ["1학년", "2학년", "3학년", "4학년", "5학년", "6학년"]
            else:
                grade_options = ["1학년", "2학년", "3학년"]
            grade = st.selectbox("학년", grade_options)
            student_name = st.text_input("학생 이름", value="")

        st.markdown("#### 상담 설정")
        counseling_issue = st.text_area(
            "상담할 주요 내용",
            placeholder="예) 학교 생활, 친구 관계, 학업 부담 등",
            height=80,
        )

        col1, col2 = st.columns(2)
        with col1:
            conversation_length = st.selectbox(
                "대화 길이 조절", ["단계 1 (짧음)", "단계 2 (보통)", "단계 3 (긴 대화)"]
            )
        with col2:
            parent_attitude = st.selectbox("학부모 상담 태도",
                                           list(PARENT_CONSULTATION_ATTITUDES.keys()),
                                           help="학부모가 상담 과정에서 보이는 주요 태도와 소통 방식을 선택하세요.")

        st.markdown("#### Big Five 성격 특성 선택(학생용)")
        # 5개 범주 각각에 대해 선택
        openness = st.selectbox("개방성 (Openness)", ["높은 개방성", "중간 개방성", "낮은 개방성"])
        conscientiousness = st.selectbox("성실성 (Conscientiousness)", ["높은 성실성", "중간 성실성", "낮은 성실성"])
        extraversion = st.selectbox("외향성 (Extraversion)", ["높은 외향성", "중간 외향성", "낮은 외향성"])
        agreeableness = st.selectbox("친화성 (Agreeableness)", ["높은 친화성", "중간 친화성", "낮은 친화성"])
        neuroticism = st.selectbox("정서적 안정성/신경증", ["높은 정서적 안정성", "중간 정서적 안정성", "높은 신경증"])
        st.markdown("""
    **추가로 Big Five 성격검사를 직접 해보시려면**  
    👉 [kakao Big Five 검사](https://together.kakao.com/big-five)를 참고해보세요.

    해당 검사를 통해 개방성, 성실성, 외향성, 친화성, 정서적 안정성(신경증) 등의
    자신의 성격 특성을 간단히 측정하고, 상담 시나리오에 현실감 있게 반영할 수 있습니다.
    """)
        # MBTI 정보
        st.markdown("#### MBTI 정보 (선택 사항)")
        col1, col2 = st.columns(2)
        with col1:
            mbti_types = ["", "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
                          "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"]
            student_mbti = st.selectbox("학생 MBTI", mbti_types)
        with col2:
            teacher_mbti = st.selectbox("선생님 MBTI", mbti_types)

        # 상담 정보 저장 버튼
        submit_info = st.form_submit_button("상담 정보 저장", use_container_width=True)

    if submit_info:
        # 5개 Big Five 범주를 dict로 저장
        big_five_traits = {
            "개방성(Openness)": openness,
            "성실성(Conscientiousness)": conscientiousness,
            "외향성(Extraversion)": extraversion,
            "친화성(Agreeableness)": agreeableness,
            "정서적 안정성/신경증(Neuroticism)": neuroticism,
        }

        st.session_state.data = {
            "school_type": school_type,
            "gender": gender,
            "grade": grade,
            "counseling_issue": counseling_issue,
            "parent_attitude": parent_attitude,
            "conversation_length": conversation_length,
            "student_name": student_name if student_name else "",
            "student_mbti": student_mbti,
            "teacher_mbti": teacher_mbti,
            "consultation_date": datetime.now().strftime("%Y-%m-%d"),
            # 여기서 중요한 변경: Big Five 전체 정보 저장
            "big_five_traits": big_five_traits
        }

        st.session_state.chat_history = []
        st.session_state.greeting_sent = False
        st.sidebar.success("✅ 상담 정보가 저장되었습니다. 대화 역할을 선택하고 채팅을 진행하세요.")

    # 역할 선택
    mode_config = {
        "선생님 -> 가상 학부모": {
            "input_avatar": "👨‍🏫",
            "response_avatar": "👨‍👩‍👧‍👦",
            "description": "선생님 역할로 학부모에게 조언이나 피드백을 제공합니다."
        },
        "학부모 -> 가상 선생님": {
            "input_avatar": "👨‍👩‍👧‍👦",
            "response_avatar": "👨‍🏫",
            "description": "학부모 역할로 선생님에게 질문하거나 상담을 받습니다."
        },
        "학생 -> 가상 선생님": {
            "input_avatar": "🧑‍🎓",
            "response_avatar": "👨‍🏫",
            "description": "학생 역할로 선생님에게 상담이나 조언을 구합니다."
        },
        "선생님 -> 가상 학생": {
            "input_avatar": "👨‍🏫",
            "response_avatar": "🧑‍🎓",
            "description": "선생님 역할로 학생과 대화하거나 조언합니다."
        },
    }

    st.markdown("## 🎭 대화 역할 선택")
    cols = st.columns(4)
    for i, (role, config) in enumerate(mode_config.items()):
        with cols[i]:
            if st.button(
                f"{config['input_avatar']} → {config['response_avatar']}\n{role}",
                key=f"role_{i}",
                help=config["description"],
                use_container_width=True,
                type="primary" if st.session_state.get("selected_role") == role else "secondary"
            ):
                st.session_state.selected_role = role
                if st.session_state.chat_history and st.session_state.chat_history[0].get("role") == "system":
                    if st.session_state.chat_history[0].get("mode") != role:
                        st.session_state.chat_history = []
                        st.session_state.greeting_sent = False
                st.rerun()

    role_mode = st.session_state.get("selected_role", list(mode_config.keys())[0])

    if not st.session_state.chat_history:
        if st.session_state.data:
            initialize_chat_history(st.session_state.data, role_mode)
            if st.session_state.chat_history and st.session_state.chat_history[0].get("role") == "system":
                st.session_state.chat_history[0]["mode"] = role_mode

    st.markdown("## 💬 상담 채팅")

    if st.session_state.data:
        with st.expander("📋 현재 상담 정보 보기", expanded=False):
            cols = st.columns([1, 1, 1])

            student_info_txt = f"{st.session_state.data.get('school_type', '')} {st.session_state.data.get('grade', '')} {st.session_state.data.get('gender', '')} {st.session_state.data.get('student_name', '')}"

            with cols[0]:
                st.markdown("##### 👨‍🎓 학생 정보")
                st.markdown(f"**기본 정보:** {student_info_txt}")

                # Big Five
                big_five_dict = st.session_state.data.get("big_five_traits", {})
                if big_five_dict:
                    st.markdown("**Big Five 성격 특성:**")
                    for cat, lvl in big_five_dict.items():
                        st.markdown(f"- {cat}: {lvl}")

                # 학생 MBTI
                if st.session_state.data.get("student_mbti"):
                    st.markdown(f"**학생 MBTI:** {st.session_state.data['student_mbti']}")

            with cols[1]:
                st.markdown("##### 👨‍🏫 교사 정보")
                if st.session_state.data.get("teacher_mbti"):
                    st.markdown(f"**교사 MBTI:** {st.session_state.data['teacher_mbti']}")
                else:
                    st.markdown("MBTI 정보가 설정되지 않았습니다.")

                st.markdown("##### 📝 상담 설정")
                st.markdown(f"**대화 길이:** {st.session_state.data.get('conversation_length', '단계 2 (보통)')}")
                st.markdown(f"**상담 주제:** {st.session_state.data.get('counseling_issue', '미지정')}")

            with cols[2]:
                st.markdown("##### 👨‍👩‍👧‍👦 학부모 정보")
                if st.session_state.data.get("parent_attitude"):
                    st.markdown(f"**상담 태도:** {st.session_state.data['parent_attitude']}")
                else:
                    st.markdown("상담 태도가 설정되지 않았습니다.")

                st.markdown("##### 🔄 현재 역할")
                st.markdown(f"**선택된 역할:** {role_mode}")

    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history[1:]:
                msg_mode = message.get("mode", role_mode)
                if message["role"] == "assistant":
                    avatar = mode_config[msg_mode]["response_avatar"]
                    st.chat_message("assistant", avatar=avatar).write(message["content"])
                elif message["role"] == "user":
                    avatar = mode_config[msg_mode]["input_avatar"]
                    st.chat_message("user", avatar=avatar).write(message["content"])

    user_input = st.chat_input("메시지를 입력하세요", key="chat_input")
    if user_input:
        st.session_state.user_input = ""
        if "teacher_suggestions" in st.session_state:
            del st.session_state.teacher_suggestions

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "mode": role_mode}
        )

        if role_mode == "선생님 -> 가상 학부모":
            with st.spinner("가상 학부모 응답 생성 중..."):
                reply = generate_parent_response(st.session_state.chat_history)
        elif role_mode == "학부모 -> 가상 선생님":
            with st.spinner("가상 선생님 응답 생성 중..."):
                reply = generate_teacher_response(st.session_state.chat_history)
        elif role_mode == "학생 -> 가상 선생님":
            with st.spinner("가상 선생님 응답 생성 중..."):
                reply = generate_teacher_response_for_student(st.session_state.chat_history)
        else:  # "선생님 -> 가상 학생"
            with st.spinner("가상 학생 응답 생성 중..."):
                reply = generate_student_response(st.session_state.chat_history)

        with st.chat_message("assistant", avatar=mode_config[role_mode]["response_avatar"]):
            placeholder = st.empty()
            streamed_text = ""
            for ch in reply:
                streamed_text += ch
                placeholder.markdown(streamed_text)
                time.sleep(0.01)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply, "mode": role_mode}
        )
        st.rerun()

    st.markdown("### 🛠️ 대화 도구")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("💡 추천 답변 보기", key="suggest_btn", use_container_width=True):
            with st.spinner("추천 답변 생성 중..."):
                if role_mode == "선생님 -> 가상 학부모":
                    suggestions = generate_teacher_input_suggestions(st.session_state.chat_history)
                elif role_mode == "학부모 -> 가상 선생님":
                    suggestions = generate_parent_input_suggestions(st.session_state.chat_history)
                elif role_mode == "학생 -> 가상 선생님":
                    suggestions = generate_student_input_suggestions(st.session_state.chat_history)
                else:
                    suggestions = generate_teacher_input_suggestions_for_student(st.session_state.chat_history)
                st.session_state.teacher_suggestions = suggestions
                st.rerun()

    with col2:
        if st.button("✅ 대화 종료", key="end_chat_btn", use_container_width=True):
            with st.spinner("대화 마무리 메시지 생성 중..."):
                closing_reply = generate_closing_message(role_mode, st.session_state.chat_history)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": closing_reply, "mode": role_mode}
            )
            st.rerun()

    with col3:
        if st.button("🗑️ 대화 초기화", key="reset_chat_btn", use_container_width=True):
            st.session_state.chat_history = []
            if "teacher_suggestions" in st.session_state:
                del st.session_state.teacher_suggestions
            st.session_state.greeting_sent = False
            st.rerun()

    if "teacher_suggestions" in st.session_state:
        display_suggestions(st.session_state.teacher_suggestions)

    with st.expander("ℹ️ 앱 정보", expanded=False):
        st.markdown("""
        ### 📚 리얼 상담 시뮬레이터
        
        이 앱은 MBTI 성격 유형, Big Five 성격 특성 모델, 교육학 이론을 통합하여 맞춤형 상담 시뮬레이션을 제공합니다. 
        교사, 학부모, 학생 간의 효과적인 의사소통을 연습하고 교육적 이론을 실제 상담에 적용하는 방법을 배울 수 있습니다.
        
        **주요 기능:**
        - MBTI 기반 성격 유형 분석 및 상담
        - Big Five 성격 특성 모델 적용 (5개 특성 모두 선택 가능)
        - 학부모 상담 태도에 따른 대응 전략
        - 교육학 이론 기반 추천 답변
        - 다양한 상담 역할 시뮬레이션
        
        © 2025 교육상담 시뮬레이션 시스템
        """)


def set_page_config():
    try:
        st.set_page_config(
            page_title="리얼 상담 시뮬레이터", page_icon="👨‍👩‍👧‍👦", layout="wide"
        )
    except Exception as e:
        st.error(f"페이지 설정 오류: {e}")

    st.markdown(
        """
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #6A82FB 0%, #FC5C7D 100%);
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-desc {
            text-align: center;
            margin-bottom: 2rem;
            font-size: 1.1rem;
            color: #555;
        }
        [data-testid="stChatMessage"] {
            max-width: 100% !important;
            width: 100% !important;
            font-size: 1.1em;
            border-radius: 15px !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .stChatMessage > div {
            padding: 15px !important;
        }
        .suggestion-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #4e8cff;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
        }
        .suggestion-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .theory-box {
            background-color: #f1f8e9;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #7cb342;
            font-size: 0.9em;
        }
        .theory-title {
            font-weight: bold;
            color: #33691e;
            margin-bottom: 5px;
        }
        .mbti-info {
            background-color: #e3f2fd;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #2196f3;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .student-info {
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #4caf50;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .teacher-info {
            background-color: #fff8e1;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #ffc107;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .parent-info {
            background-color: #f3e5f5; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 10px 0; 
            border-left: 5px solid #9c27b0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .role-button {
            border-radius: 20px !important;
            margin: 5px !important;
            background-color: #f8f9fa !important;
            border: 2px solid #e9ecef !important;
            color: #495057 !important;
            font-weight: bold !important;
            transition: all 0.3s !important;
        }
        .role-button:hover {
            background-color: #e9ecef !important;
            transform: translateY(-2px);
        }
        .role-button.active {
            border-color: #4e8cff !important;
            background-color: #e7f0ff !important;
        }
        .sidebar-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e9ecef;
        }
        .info-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 5px;
        }
        .badge-student {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .badge-teacher {
            background-color: #fff8e1;
            color: #f57f17;
        }
        .badge-parent {
            background-color: #f3e5f5;
            color: #7b1fa2;
        }
        .chat-controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .stButton>button {
            border-radius: 20px;
            padding: 5px 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-header">
            <h1>👨‍👩‍👧‍👦 리얼 상담 시뮬레이터</h1>
        </div>
        <div class="main-desc">
            <p>MBTI 성격 유형, Big Five 성격 특성, 교육학 이론을 활용한 맞춤형 상담 시뮬레이션 도구입니다.<br>
            다양한 역할과 상황에서 효과적인 상담 기술을 연습해보세요.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
