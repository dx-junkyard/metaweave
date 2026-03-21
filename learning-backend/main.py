"""MetaWeave Learning Backend — 学習UI専用のAPIサーバー。

独立したコンテナで動作し、コース管理・RAGチャット・進捗追跡を担当する。
認証はメインbackendと同じJWTシークレットを共有して検証する。

Endpoints
---------
POST /api/learning/courses                              コースを新規作成
GET  /api/learning/courses                              コース一覧
GET  /api/learning/courses/{course_id}                  コース詳細
PUT  /api/learning/courses/{course_id}                  コースを更新
DELETE /api/learning/courses/{course_id}                 コースを削除
GET  /api/learning/courses/{course_id}/progress          進捗データ
GET  /api/learning/courses/{cid}/topics/{tid}/chat       チャット履歴
POST /api/learning/courses/{cid}/topics/{tid}/chat       RAGチャット
GET  /healthz                                            ヘルスチェック
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from functools import lru_cache

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from neo4j import GraphDatabase
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (環境変数から読み込み)
# ---------------------------------------------------------------------------

_JWT_SECRET: str = os.environ.get("JWT_SECRET", "metaweave-dev-secret-change-in-prod")
_JWT_ALGORITHM: str = "HS256"

_QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "qdrant")
_QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
_QDRANT_COLLECTION: str = "papers"
_VECTOR_DIM: int = 3072

_NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
_NEO4J_AUTH_STR: str = os.environ.get("NEO4J_AUTH", "neo4j/metaweave")

_OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
_OPENAI_ANALYSIS_MODEL: str = os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o")
_OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_bearer = HTTPBearer()

app = FastAPI(title="MetaWeave Learning API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _neo4j_driver():
    user, password = _NEO4J_AUTH_STR.split("/", 1)
    return GraphDatabase.driver(_NEO4J_URI, auth=(user, password))


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    return QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)


@lru_cache(maxsize=1)
def _openai() -> OpenAI:
    return OpenAI(api_key=_OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Auth (メインbackendと同じJWTを検証)
# ---------------------------------------------------------------------------

def _get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Bearer トークンをデコードしてユーザー情報を返す。"""
    try:
        payload = jwt.decode(
            credentials.credentials, _JWT_SECRET, algorithms=[_JWT_ALGORITHM]
        )
        return {
            "id": payload["sub"],
            "username": payload["username"],
            "email": payload["email"],
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LearningPrerequisite(BaseModel):
    name: str
    status: str = "not_started"  # mastered | partial | not_started


class LearningMisconception(BaseModel):
    label: str = ""
    wrong: str
    correct: str


class LearningTopic(BaseModel):
    id: str
    title: str
    chapter_index: int
    status: str = "locked"  # completed | in_progress | locked
    prerequisites: list[LearningPrerequisite] = []
    misconceptions: list[LearningMisconception] = []


class LearningChapter(BaseModel):
    title: str
    status: str = "locked"  # completed | in_progress | locked
    progress_pct: int = 0


class LearningConcept(BaseModel):
    name: str
    status: str = "future"  # mastered | learning | future
    children: list[str] = []
    expanded: bool = False


class LearningSource(BaseModel):
    title: str
    subtitle: str = ""
    license: str = ""
    used_section: str = ""
    arxiv_id: str = ""  # MetaWeaveの論文と紐付ける場合


class LearningReferencedSection(BaseModel):
    source: str
    section: str
    title: str
    note: str = ""


class CourseCreateRequest(BaseModel):
    """コース新規作成リクエスト。"""

    title: str
    chapters: list[LearningChapter] = []
    topics: list[LearningTopic] = []
    concepts: list[LearningConcept] = []
    sources: list[LearningSource] = []


class CourseUpdateRequest(BaseModel):
    """コース更新リクエスト。部分更新に対応。"""

    title: str | None = None
    chapters: list[LearningChapter] | None = None
    topics: list[LearningTopic] | None = None
    concepts: list[LearningConcept] | None = None
    sources: list[LearningSource] | None = None


class LearningCourseOut(BaseModel):
    id: str
    title: str


class LearningCourseDetail(BaseModel):
    id: str
    title: str
    chapters: list[LearningChapter] = []
    topics: list[LearningTopic] = []
    concepts: list[LearningConcept] = []
    sources: list[LearningSource] = []
    referenced_sections: list[LearningReferencedSection] = []
    progress: dict | None = None


class LearningSession(BaseModel):
    date: str
    topic: str
    duration: str


class LearningProgress(BaseModel):
    mastered_concepts: int = 0
    learning_concepts: int = 0
    misconceptions: int = 0
    streak_days: int = 0
    sessions: list[LearningSession] = []


class LearningChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class LearningChatResponse(BaseModel):
    answer: str
    course_update: dict | None = None


class LearningChatHistoryResponse(BaseModel):
    history: list[dict]


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _get_course_data(user_id: str, course_id: str) -> dict | None:
    """Neo4j から LearningCourse データを取得する。"""
    driver = _neo4j_driver()
    with driver.session() as session:
        record = session.run(
            """
            MATCH (u:User {id: $user_id})-[:ENROLLED_IN]->(lc:LearningCourse {id: $course_id})
            RETURN lc.data AS data
            """,
            user_id=user_id,
            course_id=course_id,
        ).single()
        if record and record["data"]:
            try:
                return json.loads(record["data"])
            except Exception:
                return None
    return None


def _save_course_data(user_id: str, course_id: str, data: dict) -> None:
    """LearningCourse データを Neo4j に永続化する。"""
    driver = _neo4j_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            MERGE (lc:LearningCourse {id: $course_id})
            MERGE (u)-[:ENROLLED_IN]->(lc)
            SET lc.data = $data, lc.updated_at = $now
            """,
            user_id=user_id,
            course_id=course_id,
            data=json.dumps(data, ensure_ascii=False),
            now=datetime.datetime.utcnow().isoformat(),
        )


def _delete_course_data(user_id: str, course_id: str) -> bool:
    """LearningCourse ノードとリレーションを削除する。"""
    driver = _neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (u:User {id: $user_id})-[r:ENROLLED_IN]->(lc:LearningCourse {id: $course_id})
            DELETE r
            WITH lc
            OPTIONAL MATCH (lc)<-[:ENROLLED_IN]-(:User)
            WITH lc, count(*) AS remaining
            WHERE remaining = 0
            DELETE lc
            RETURN true AS deleted
            """,
            user_id=user_id,
            course_id=course_id,
        ).single()
        return result is not None


# ---------------------------------------------------------------------------
# RAG helpers (Qdrant + OpenAI Embedding)
# ---------------------------------------------------------------------------

def _embed_text(text: str) -> list[float]:
    """テキストを embedding ベクトルに変換する。"""
    client = _openai()
    resp = client.embeddings.create(model=_OPENAI_EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def _search_relevant_chunks(
    query: str,
    arxiv_ids: list[str],
    top_k: int = 5,
) -> list[str]:
    """Qdrant から関連チャンクをベクトル検索する。

    コースに紐づいた論文(arxiv_ids)のチャンクから、質問に最も近いものを返す。
    arxiv_ids が空の場合はフィルタなしで全体から検索する。
    """
    if not arxiv_ids:
        return []

    try:
        query_vector = _embed_text(query)
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return []

    try:
        # 複数の arxiv_id に跨って検索
        must_conditions = []
        if len(arxiv_ids) == 1:
            must_conditions.append(
                FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_ids[0]))
            )
        else:
            # Qdrant の should で OR 検索
            from qdrant_client.models import Filter as QFilter
            should_conditions = [
                FieldCondition(key="arxiv_id", match=MatchValue(value=aid))
                for aid in arxiv_ids
            ]
            result = _qdrant().query_points(
                collection_name=_QDRANT_COLLECTION,
                query=query_vector,
                query_filter=QFilter(should=should_conditions),
                limit=top_k,
                with_payload=True,
            )
            return [
                hit.payload.get("text", "")
                for hit in result.points
                if hit.payload
            ]

        result = _qdrant().query_points(
            collection_name=_QDRANT_COLLECTION,
            query=query_vector,
            query_filter=Filter(must=must_conditions),
            limit=top_k,
            with_payload=True,
        )
        return [
            hit.payload.get("text", "")
            for hit in result.points
            if hit.payload
        ]
    except Exception as exc:
        logger.warning("Qdrant search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Progress calculation
# ---------------------------------------------------------------------------

def _calculate_progress(user_id: str, course_id: str, course_data: dict) -> dict:
    """コースデータとチャット履歴から進捗を計算する。"""
    topics = course_data.get("topics", [])
    concepts = course_data.get("concepts", [])

    mastered = sum(1 for c in concepts if c.get("status") == "mastered")
    learning = sum(1 for c in concepts if c.get("status") == "learning")

    total_misconceptions = 0
    for t in topics:
        total_misconceptions += len(t.get("misconceptions", []))

    # セッション履歴を Neo4j のチャット履歴から構築
    sessions = []
    driver = _neo4j_driver()
    with driver.session() as session:
        records = session.run(
            """
            MATCH (u:User {id: $user_id})-[r:LEARNING_CHAT]->(lt:LearningTopic {course_id: $course_id})
            RETURN lt.id AS topic_id, r.history AS history, r.updated_at AS updated_at
            ORDER BY r.updated_at DESC
            LIMIT 10
            """,
            user_id=user_id,
            course_id=course_id,
        ).data()

    for r in records:
        history = []
        if r.get("history"):
            try:
                history = json.loads(r["history"])
            except Exception:
                pass

        # トピック名を取得
        topic_name = r["topic_id"]
        for t in topics:
            if t.get("id") == r["topic_id"]:
                topic_name = t.get("title", topic_name)
                break

        # メッセージ数からおおよその時間を推定 (1メッセージ≒2分)
        msg_count = len(history)
        duration_min = max(5, msg_count * 2)

        date_str = ""
        if r.get("updated_at"):
            try:
                dt = datetime.datetime.fromisoformat(r["updated_at"])
                date_str = f"{dt.month}/{dt.day}"
            except Exception:
                pass

        sessions.append({
            "date": date_str or "---",
            "topic": topic_name,
            "duration": f"{duration_min}分",
        })

    # 連続学習日数を計算
    streak = _calculate_streak(user_id, course_id)

    return {
        "mastered_concepts": mastered,
        "learning_concepts": learning,
        "misconceptions": total_misconceptions,
        "streak_days": streak,
        "sessions": sessions[:5],
    }


def _calculate_streak(user_id: str, course_id: str) -> int:
    """チャット履歴の日付から連続学習日数を算出する。"""
    driver = _neo4j_driver()
    with driver.session() as session:
        records = session.run(
            """
            MATCH (u:User {id: $user_id})-[r:LEARNING_CHAT]->(lt:LearningTopic {course_id: $course_id})
            WHERE r.updated_at IS NOT NULL
            RETURN DISTINCT r.updated_at AS updated_at
            ORDER BY r.updated_at DESC
            """,
            user_id=user_id,
            course_id=course_id,
        ).data()

    if not records:
        return 0

    dates = set()
    for r in records:
        try:
            dt = datetime.datetime.fromisoformat(r["updated_at"])
            dates.add(dt.date())
        except Exception:
            continue

    if not dates:
        return 0

    sorted_dates = sorted(dates, reverse=True)
    today = datetime.date.today()

    # 今日または昨日から連続している日数を数える
    if sorted_dates[0] < today - datetime.timedelta(days=1):
        return 0

    streak = 1
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] == sorted_dates[i - 1] - datetime.timedelta(days=1):
            streak += 1
        else:
            break

    return streak


# ---------------------------------------------------------------------------
# Course CRUD endpoints
# ---------------------------------------------------------------------------

@app.post("/api/learning/courses", response_model=LearningCourseOut, status_code=201)
def create_course(
    body: CourseCreateRequest,
    current_user: dict = Depends(_get_current_user),
) -> LearningCourseOut:
    """新しいコースを作成する。

    JSON で章構成・トピック・概念マップ・教材リストを登録する。
    """
    import uuid
    course_id = str(uuid.uuid4())[:8]

    data = {
        "id": course_id,
        "title": body.title,
        "chapters": [ch.model_dump() for ch in body.chapters],
        "topics": [t.model_dump() for t in body.topics],
        "concepts": [c.model_dump() for c in body.concepts],
        "sources": [s.model_dump() for s in body.sources],
        "referenced_sections": [],
    }

    _save_course_data(current_user["id"], course_id, data)
    logger.info("Created course '%s' (id=%s) for user=%s", body.title, course_id, current_user["id"])

    return LearningCourseOut(id=course_id, title=body.title)


@app.get("/api/learning/courses", response_model=list[LearningCourseOut])
def list_courses(
    current_user: dict = Depends(_get_current_user),
) -> list[LearningCourseOut]:
    """ユーザーが登録しているコース一覧を返す。"""
    driver = _neo4j_driver()
    with driver.session() as session:
        records = session.run(
            """
            MATCH (u:User {id: $user_id})-[:ENROLLED_IN]->(lc:LearningCourse)
            RETURN lc.id AS id, lc.data AS data
            """,
            user_id=current_user["id"],
        ).data()

    courses = []
    for r in records:
        data = {}
        if r.get("data"):
            try:
                data = json.loads(r["data"])
            except Exception:
                pass
        courses.append(LearningCourseOut(
            id=r["id"],
            title=data.get("title", r["id"]),
        ))

    return courses


@app.get("/api/learning/courses/{course_id}", response_model=LearningCourseDetail)
def get_course(
    course_id: str,
    current_user: dict = Depends(_get_current_user),
) -> LearningCourseDetail:
    """コースの詳細データを返す。"""
    data = _get_course_data(current_user["id"], course_id)
    if not data:
        raise HTTPException(status_code=404, detail="Course not found")

    return LearningCourseDetail(**data)


@app.put("/api/learning/courses/{course_id}", response_model=LearningCourseDetail)
def update_course(
    course_id: str,
    body: CourseUpdateRequest,
    current_user: dict = Depends(_get_current_user),
) -> LearningCourseDetail:
    """コースを部分更新する。指定されたフィールドのみ上書き。"""
    data = _get_course_data(current_user["id"], course_id)
    if not data:
        raise HTTPException(status_code=404, detail="Course not found")

    if body.title is not None:
        data["title"] = body.title
    if body.chapters is not None:
        data["chapters"] = [ch.model_dump() for ch in body.chapters]
    if body.topics is not None:
        data["topics"] = [t.model_dump() for t in body.topics]
    if body.concepts is not None:
        data["concepts"] = [c.model_dump() for c in body.concepts]
    if body.sources is not None:
        data["sources"] = [s.model_dump() for s in body.sources]

    _save_course_data(current_user["id"], course_id, data)
    logger.info("Updated course %s for user=%s", course_id, current_user["id"])

    return LearningCourseDetail(**data)


@app.delete("/api/learning/courses/{course_id}", status_code=204)
def delete_course(
    course_id: str,
    current_user: dict = Depends(_get_current_user),
) -> None:
    """コースを削除する。"""
    deleted = _delete_course_data(current_user["id"], course_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Course not found")

    logger.info("Deleted course %s for user=%s", course_id, current_user["id"])


# ---------------------------------------------------------------------------
# Progress endpoint
# ---------------------------------------------------------------------------

@app.get("/api/learning/courses/{course_id}/progress", response_model=LearningProgress)
def get_progress(
    course_id: str,
    current_user: dict = Depends(_get_current_user),
) -> LearningProgress:
    """コースの進捗データを計算して返す。"""
    data = _get_course_data(current_user["id"], course_id)
    if not data:
        raise HTTPException(status_code=404, detail="Course not found")

    progress = _calculate_progress(current_user["id"], course_id, data)
    return LearningProgress(**progress)


# ---------------------------------------------------------------------------
# Chat endpoints (RAG統合)
# ---------------------------------------------------------------------------

@app.get(
    "/api/learning/courses/{course_id}/topics/{topic_id}/chat",
    response_model=LearningChatHistoryResponse,
)
def get_chat_history(
    course_id: str,
    topic_id: str,
    current_user: dict = Depends(_get_current_user),
) -> LearningChatHistoryResponse:
    """トピックのチャット履歴を返す。"""
    driver = _neo4j_driver()
    with driver.session() as session:
        record = session.run(
            """
            MATCH (u:User {id: $user_id})-[r:LEARNING_CHAT]->(lt:LearningTopic {id: $topic_id, course_id: $course_id})
            RETURN r.history AS history
            """,
            user_id=current_user["id"],
            topic_id=topic_id,
            course_id=course_id,
        ).single()

    if not record or not record.get("history"):
        return LearningChatHistoryResponse(history=[])

    try:
        history = json.loads(record["history"])
    except Exception:
        history = []

    return LearningChatHistoryResponse(history=history)


_LEARNING_SYSTEM_PROMPT = """あなたは学習者の深い理解を支援する家庭教師です。
以下の原則に従ってください。

**教育方針:**
1. 学生の誤解を発見したら「訂正：」と明記し、なぜその誤解が生じやすいか説明してください。
2. 概念の説明は具体的な数式、図、または例を使って行ってください。
3. 教材から引用できる場合は出典（セクション番号等）を明記してください。
4. 説明の最後に、理解を確認するための質問をしてください。
5. 関連する概念へのドリルダウン選択肢を提示してください。

**RAGコンテキスト利用:**
- 提供される「教材チャンク」はベクトル検索で取得した関連箇所です。
- これらを根拠として回答し、出典を明記してください。
- コンテキストに含まれない情報について推測する場合はその旨を明記してください。

**フォーマット:**
- 誤解の訂正が必要な場合は最初に「訂正：」と記述
- 参照した教材のセクションがあれば言及
- 回答の末尾に深掘りできるトピックを `[〇〇について詳しく聞く]` の形式で提示"""


@app.post(
    "/api/learning/courses/{course_id}/topics/{topic_id}/chat",
    response_model=LearningChatResponse,
)
def learning_chat(
    course_id: str,
    topic_id: str,
    body: LearningChatRequest,
    current_user: dict = Depends(_get_current_user),
) -> LearningChatResponse:
    """RAG統合された学習チャットエンドポイント。

    1. コースに紐づいた論文の arxiv_id を収集
    2. Qdrant でユーザーの質問に関連するチャンクをベクトル検索
    3. チャンク + コース情報 + 履歴をプロンプトに組み込んで LLM に回答生成
    4. 応答から誤解検出を行い、コースデータを更新
    """
    # 1. コースデータを取得
    course_data = _get_course_data(current_user["id"], course_id)
    if not course_data:
        raise HTTPException(status_code=404, detail="Course not found")

    # トピック情報を取得
    topic_info = None
    for t in course_data.get("topics", []):
        if t.get("id") == topic_id:
            topic_info = t
            break
    topic_title = topic_info["title"] if topic_info else topic_id

    # 2. RAG: コースの教材に紐づいた arxiv_id を収集してチャンク検索
    arxiv_ids = [
        s["arxiv_id"]
        for s in course_data.get("sources", [])
        if s.get("arxiv_id")
    ]
    relevant_chunks = _search_relevant_chunks(body.message, arxiv_ids, top_k=5)

    # 3. コンテキストブロックを構築
    context_parts: list[str] = []
    if relevant_chunks:
        context_parts.append(
            "## 教材から検索された関連箇所\n" + "\n---\n".join(relevant_chunks)
        )

    # コースのソース情報もコンテキストに含める
    sources_info = []
    for s in course_data.get("sources", []):
        info = s.get("title", "")
        if s.get("subtitle"):
            info += f" — {s['subtitle']}"
        sources_info.append(info)
    if sources_info:
        context_parts.append("## 登録済み教材\n" + "\n".join(f"- {s}" for s in sources_info))

    context_block = "\n\n".join(context_parts) if context_parts else "(教材コンテキストなし)"

    # 4. LLM メッセージ構築
    course_title = course_data.get("title", course_id)
    messages: list[dict] = [
        {"role": "system", "content": _LEARNING_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"コース: {course_title}\n"
            f"現在のトピック: {topic_title}\n\n"
            f"{context_block}\n\n"
            "上記のコンテキストを念頭に置いて質問に回答してください。"
        )},
        {"role": "assistant", "content": (
            f"了解しました。「{topic_title}」について、教材を参照しながら学習を進めましょう。"
        )},
    ]

    for turn in body.history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": body.message})

    # 5. LLM 呼び出し
    try:
        client = _openai()
        response = client.chat.completions.create(
            model=_OPENAI_ANALYSIS_MODEL,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("Learning chat LLM call failed for topic %s", topic_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    # 6. 誤解検出: LLMの応答に「訂正」が含まれていたらコースデータに追記
    course_update = None
    if topic_info and "訂正" in answer:
        course_update = _detect_and_record_misconception(
            current_user["id"], course_id, course_data, topic_id, body.message, answer
        )

    # 7. チャット履歴を永続化
    updated_history = body.history + [
        {"role": "user", "content": body.message},
        {"role": "assistant", "content": answer},
    ]
    try:
        driver = _neo4j_driver()
        with driver.session() as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                MERGE (lt:LearningTopic {id: $topic_id, course_id: $course_id})
                MERGE (u)-[r:LEARNING_CHAT]->(lt)
                SET r.history = $history, r.updated_at = $now
                """,
                user_id=current_user["id"],
                topic_id=topic_id,
                course_id=course_id,
                history=json.dumps(updated_history, ensure_ascii=False),
                now=datetime.datetime.utcnow().isoformat(),
            )
    except Exception:
        logger.exception(
            "Failed to persist learning chat for user=%s topic=%s",
            current_user["id"], topic_id,
        )

    return LearningChatResponse(answer=answer, course_update=course_update)


# ---------------------------------------------------------------------------
# Misconception detection
# ---------------------------------------------------------------------------

def _detect_and_record_misconception(
    user_id: str,
    course_id: str,
    course_data: dict,
    topic_id: str,
    user_message: str,
    ai_response: str,
) -> dict | None:
    """AI応答から誤解を検出し、コースデータに記録する。

    応答に「訂正」が含まれている場合、ユーザーの発言を「誤解」、
    AIの訂正内容を「正しい理解」として記録する。
    """
    # ユーザーの発言を短縮して誤解ラベルにする
    wrong = user_message
    if len(wrong) > 60:
        wrong = wrong[:60] + "…"

    # 訂正部分を抽出（「訂正：」以降の最初の段落）
    correct = ""
    for line in ai_response.split("\n"):
        if "訂正" in line:
            # 「訂正：」の後のテキストを取得
            idx = line.find("訂正")
            rest = line[idx:]
            # 「訂正：」や「訂正」の後のテキスト
            for sep in ["：", ":", "】"]:
                if sep in rest:
                    correct = rest.split(sep, 1)[1].strip()
                    break
            if not correct:
                correct = rest.replace("訂正", "").strip()
            break

    if not correct:
        correct = "（AIの応答を参照してください）"

    today = datetime.date.today()
    misconception = {
        "label": f"{today.month}/{today.day} の訂正",
        "wrong": wrong,
        "correct": correct,
    }

    # コースデータのトピックに誤解を追記
    for t in course_data.get("topics", []):
        if t.get("id") == topic_id:
            if "misconceptions" not in t:
                t["misconceptions"] = []
            t["misconceptions"].insert(0, misconception)
            # 最新5件のみ保持
            t["misconceptions"] = t["misconceptions"][:5]
            break

    _save_course_data(user_id, course_id, course_data)

    # フロントエンドに更新を通知
    return {
        "topics": course_data.get("topics", []),
        "concepts": course_data.get("concepts", []),
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "learning-backend"}
