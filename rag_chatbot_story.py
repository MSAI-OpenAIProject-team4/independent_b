# rag_chatbot.py

import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI

# ✅ Azure OpenAI 설정
chat_client = AzureOpenAI(
  api_key="보안",
  api_version="보안",
  azure_endpoint="보안"
)

embed_client = AzureOpenAI(
  api_key = "보안",
  api_version = "보안",
  azure_endpoint = "보안"
)

EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"


### ✅ 1. CSV 로딩 및 전처리
def load_embedded_csv(filepath):
    df = pd.read_csv(filepath)
    df['ada_v2'] = df['ada_v2'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    for col in ['name', 'orders', 'movementFamily', 'activities', 'content']:
        df[col] = df[col].fillna("").astype(str)
    return df

### ✅ 2. 질문 임베딩 생성
def generate_embeddings(text, model=EMBED_MODEL):
    response = embed_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

### ✅ 3. 필터링 로직
def filter_df_by_query(query, df):
    query = query.replace(" ", "").lower()
    name_filtered = df[df['name'].str.replace(" ", "").str.lower().str.contains(query, na=False)]
    if not name_filtered.empty:
        print("✅ 이름 필터링 적용:", name_filtered['name'].tolist())
        return name_filtered

    multi_filtered = df[
        df['orders'].str.lower().str.contains(query, na=False) |
        df['movementFamily'].str.lower().str.contains(query, na=False) |
        df['activities'].str.lower().str.contains(query, na=False) |
        df['content'].str.lower().str.contains(query, na=False)
    ]
    if not multi_filtered.empty:
        print("✅ 다중 필드 필터링 적용")
        return multi_filtered

    print("⚠️ 필터링 실패 → 전체 문서 사용")
    return df

### ✅ 4. 유사 문서 추출
def retrieve_relevant_context(query, df, top_k=5, embed_model=EMBED_MODEL):
    query_emb = generate_embeddings(query, model=embed_model)
    df['score'] = df['ada_v2'].apply(lambda x: cosine_similarity([x], [query_emb])[0][0])
    top_docs = df.sort_values(by="score", ascending=False).head(top_k).reset_index()

    context_blocks = []
    citations = []

    print(f"\n🔍 유사도 상위 문서 (Top {top_k}):")
    for i, row in top_docs.iterrows():
        ref_id = f"[{i+1}]"
        name = row.get("name", "")
        score = round(row["score"], 4)
        print(f"{ref_id} {name} (score: {score})")

        text = f"""{ref_id}
이름: {name}
출생지: {row.get("addressBirth", "")}
상훈: {row.get("orders", "")}
활동 내용: {row.get("activities", "")}
문서 내용: {row.get("content", "")}
"""
        context_blocks.append(text)
        citations.append({
            "index": i + 1,
            "title": name,
            "reference": row.get("references", "")
        })

    context = "\n\n".join(context_blocks)
    return context, citations

### ✅ 5. GPT 호출
def ask_gpt(query, context, chat_history, model=CHAT_MODEL, system_prompt=None):
    if system_prompt:
        chat_history.insert(0, {"role": "system", "content": system_prompt})
    chat_history.append({"role": "user", "content": f"질문: {query}\n\n아래 문서를 참고해서 대답해줘:\n\n{context}"})

    response = chat_client.chat.completions.create(
        model=model,
        messages=chat_history
    )
    reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": reply})
    return reply

### ✅ 6. 메인 실행
def main():
    df = load_embedded_csv("임베딩_완료2.csv")
    chat_history = []

    system_prompt = """
당신은 "꼬꼬무 스타일의 이야기 전달자"입니다.  
사용자가 입력한 독립운동가의 정보를 바탕으로,  
감정적으로 공감할 수 있는 대화형 스토리텔링을 제공합니다.
 
당신의 목적은 다음과 같습니다:
 
1. 사용자가 인물의 삶에 **몰입**하도록 유도하고,
2. 각 이야기 구성 단계 사이마다 **사용자에게 직접 질문을 던지고**, 
3. **사용자의 답변을 듣고 반응하며** 이야기를 이어가는 것입니다.
 
---
 
이야기는 아래 5단계로 구성되며, 각 단계마다 사용자와 **자연스러운 대화**를 주고받아야 합니다.
 
### 📌 [단계별 규칙]
 
---
 
** 프롤로그 (30~50자) **
- 진입 전, 감정 공감형 질문을 사용자에게 던지고 응답을 기다리세요.  
  예: “사랑하는 사람을 지키기 위해 모든 걸 내려놔 본 적 있으세요?”  
- 사용자의 반응에 감정적으로 공감하고,  
- 궁금증을 자극하는 한 문장으로 프롤로그를 제시하세요.  
  예: “누군가는 입을 다물었고, 누군가는 외쳤습니다. 그리고 그녀는… 소리치기로 했습니다.”
 
---
 
** 기 (사건 배경 설명) **
- 인물의 시대적 상황과 배경을 전달합니다.  
- 전달 후, 사용자가 그 시대였다면 어떤 선택을 했을지 질문하세요.  
  예: “그런 시대였다면, 당신은 어떻게 했을 것 같나요?”
 
---
 
** 승 (사건 전개) **
- 인물이 실제로 어떤 행동을 했는지 서술합니다.  
- 중간에 질문을 넣어 몰입을 유도하세요.  
  예: “당신이라면… 고문을 당하고도 다시 싸울 수 있을까요?”
 
---
 
** 전 (반전/진실) **
- 감춰졌던 진실, 고통스러운 사실, 감동적인 반전을 전합니다.  
- 전달 후, 사용자의 감정 반응을 물어보세요.  
  예: “이 사실을 듣고, 어떤 감정이 드세요?”
 
---
 
** 결 (여운과 마무리) **
- 인물의 마지막 여정과 우리가 왜 이 이야기를 기억해야 하는지를 전합니다.  
- 마지막으로, 사용자가 스스로 질문을 받아들이도록 유도하세요.  
  예: “우리는 왜, 이 이야기를 기억해야 할까요?”
 
---
 
### 💡 기타 규칙
 
- 항상 **친근한 말투**, **대화하듯 자연스럽게** 이야기하세요.  
- **질문 후에는 반드시 사용자 응답을 기다리는 멘트**를 포함하세요.  
- 사용자 반응에 따라 **진심 어린 감정 반응**을 보여주세요.  
- 너무 긴 설명 없이, **질문-응답-진행** 흐름을 지켜주세요.
- 사용자가 기에서 대답을 했으면 그 다음에는 승으로 승에서 사용자가 대답하면 승에서 전으로 전에서 사용자가 대답하면 전에서 결로 차례대로 넘아가세요.
- 유사도 높은 인물 중심으로 내용을 대답하세요.
- 문서 중 '관련 인물', '활동 내역'에서 연결점을 찾으세요.
- 사용자가 물어본 인물과 연관된 다른 인물이 있다면 이름을 명확히 알려주세요.
- 결까지 질문에 대한 대답을 마치면, 그 인물과 관련한 '인물' 혹은 '사건'을 사용자에게 사용자가 'yes' 혹은 'no'로 대답할 수 있도록 제안해주세요.
- 사용자가 'no'라고 대답하면 대화를 종료하고, 'yes'이라고 대답하면 제안사항과 유사도가 가장 높은 정보를 사용자에게 제공하세요.
- 만약에 대화 중 사용자가 5단계 대화 흐름과 다른 질문을 하면, 다른 질문에 맞는 대답을 하세요. 그때부터는 5단계 흐름이 아닌 대화형으로 사용자와 이야기를 주고 받으세요.
- 사용자와 자연스러운 대화형으로 대화가 시작되면 다시 5단계 대화 흐름으로 돌아가지 않도록 하세요.
- 사건과 인물을 함께 제안할 때 사건과 인물 전부 사용자에게 정보를 제공하도록하세요.



- 반드시 아래 제공된 문서들만 참고해서 대답하세요.
- 문서에 없는 정보는 "문서에 해당 내용이 없습니다."라고 분명히 말해주세요.
- 추론하거나 외부 지식을 추가하지 마세요.
- 사용자의 질문이 인물, 활동, 상훈, 소속 단체, 관련 인물에 대한 것이면 해당 문서 필드를 찾아서 설명해주세요.
    """

    print("💬 독립운동 RAG 챗봇입니다. '종료'라고 입력하면 종료됩니다.")

    while True:
        query = input("\n👤 질문: ").strip()

        if query.lower() in ['종료', 'quit', 'exit']:
            print("👋 챗봇을 종료합니다. 감사합니다!")
            break

        if not query:
            print("⚠️ 질문을 입력해주세요.")
            continue

        try:
            filtered_df = filter_df_by_query(query, df)
            context, citations = retrieve_relevant_context(query, filtered_df, top_k=5)
            response = ask_gpt(query, context, chat_history, model=CHAT_MODEL, system_prompt=system_prompt)

            print("\n🤖 GPT 응답:\n")
            print(response)

            print("\n📚 Citations:")
            for c in citations:
                print(f"[{c['index']}] {c['title']}\n출처: {c['reference']}\n")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
