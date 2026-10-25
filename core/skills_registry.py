"""Skills Registry - ClawHub-like tool sharing for Jarvis.

A registry for discovering, sharing, and reusing tools, prompts, and
agent capabilities across Jarvis instances and users.

Key features:
- Register and discover tools/skills
- Version management and dependencies
- Rating and reputation system
- Categories and tags for organization
- Remote skill import/export
- Permission-based access control
- Usage tracking and analytics

Example:
    registry = get_skills_registry()
    
    # Register a skill
    skill_id = registry.register(
        name="summarize_webpage",
        description="Extract and summarize content from URLs",
        category="web",
        tags=["summarization", "web", "extraction"],
        author="user_123",
        skill_type="tool",
        code="...",
    )
    
    # Search for skills
    results = registry.search("summarize", tags=["web"])
    
    # Get a skill
    skill = registry.get(skill_id)
    
    # Rate a skill
    registry.rate(skill_id, rating=5, feedback="Very useful!")
"""
import hashlib
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import secrets
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SkillType(Enum):
    TOOL = "tool"
    PROMPT = "prompt"
    PIPELINE = "pipeline"
    AGENT = "agent"
    WORKFLOW = "workflow"


class SkillStatus(Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


class AccessLevel(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    SHARED = "shared"


@dataclass
class SkillVersion:
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    changes: str = ""
    checksum: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "changes": self.changes,
            "checksum": self.checksum,
        }


@dataclass
class SkillRating:
    rating: float = 0.0
    count: int = 0
    total_score: int = 0
    feedback: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_rating(self, rating: int, feedback: str = "", user_id: str = "") -> None:
        self.total_score += rating
        self.count += 1
        self.rating = self.total_score / self.count
        if feedback:
            self.feedback.append({
                "rating": rating,
                "feedback": feedback,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
            })

    def to_dict(self) -> dict:
        return {
            "rating": round(self.rating, 2),
            "count": self.count,
            "feedback_count": len(self.feedback),
        }


@dataclass
class SkillUsage:
    total_uses: int = 0
    daily_uses: dict[str, int] = field(default_factory=dict)
    last_used: datetime | None = None
    unique_users: set[str] = field(default_factory=set)

    def record_use(self, user_id: str = "") -> None:
        self.total_uses += 1
        self.last_used = datetime.now()
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_uses[today] = self.daily_uses.get(today, 0) + 1
        if user_id:
            self.unique_users.add(user_id)

        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        old_days = [d for d in self.daily_uses if d < cutoff]
        for day in old_days:
            del self.daily_uses[day]

    def to_dict(self) -> dict:
        return {
            "total_uses": self.total_uses,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "unique_users": len(self.unique_users),
        }


@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    skill_type: SkillType
    author: str
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    status: SkillStatus = SkillStatus.DRAFT
    access: AccessLevel = AccessLevel.PUBLIC

    code: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    version: str = "1.0.0"
    versions: list[SkillVersion] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    rating: SkillRating = field(default_factory=SkillRating)
    usage: SkillUsage = field(default_factory=SkillUsage)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    published_at: datetime | None = None

    shared_with: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_code: bool = False) -> dict:
        result = {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type.value,
            "author": self.author,
            "category": self.category,
            "tags": self.tags,
            "status": self.status.value,
            "access": self.access.value,
            "version": self.version,
            "rating": self.rating.to_dict(),
            "usage": self.usage.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "dependencies": self.dependencies,
        }
        if include_code:
            result["code"] = self.code
            result["config"] = self.config
            result["parameters"] = self.parameters
        return result


@dataclass
class Category:
    category_id: str
    name: str
    description: str = ""
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    skill_count: int = 0

    def to_dict(self) -> dict:
        return {
            "category_id": self.category_id,
            "name": self.name,
            "description": self.description,
            "parent": self.parent,
            "children": self.children,
            "skill_count": self.skill_count,
        }


class SkillsRegistry:
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "skills_registry"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._skills: dict[str, Skill] = {}
        self._categories: dict[str, Category] = {}
        self._author_skills: dict[str, set[str]] = defaultdict(set)
        self._tag_index: dict[str, set[str]] = defaultdict(set)
        self._category_index: dict[str, set[str]] = defaultdict(set)
        self._type_index: dict[SkillType, set[str]] = defaultdict(set)

        self._lock = threading.RLock()
        self._max_skills = 10000
        self._max_feedback_per_skill = 100

        self._load_state()
        self._init_default_categories()

    def _init_default_categories(self) -> None:
        defaults = [
            ("general", "General Purpose", None),
            ("web", "Web & Internet", None),
            ("data", "Data Processing", None),
            ("ai", "AI & Machine Learning", None),
            ("productivity", "Productivity", None),
            ("automation", "Automation", None),
            ("security", "Security", None),
            ("system", "System Tools", None),
            ("communication", "Communication", None),
            ("creative", "Creative & Content", None),
        ]

        for cat_id, name, parent in defaults:
            if cat_id not in self._categories:
                self._categories[cat_id] = Category(
                    category_id=cat_id,
                    name=name,
                    parent=parent,
                )

    def _load_state(self) -> None:
        skills_file = self.storage_path / "skills.json"
        if skills_file.exists():
            try:
                data = read_json_with_integrity(skills_file)
                for skill_data in data.get("skills", []):
                    skill = self._deserialize_skill(skill_data)
                    self._skills[skill.skill_id] = skill
                    self._index_skill(skill)
                logger.info(f"Loaded {len(self._skills)} skills")
            except Exception as e:
                logger.warning(f"Failed to load skills: {e}")

        categories_file = self.storage_path / "categories.json"
        if categories_file.exists():
            try:
                data = read_json_with_integrity(categories_file)
                for cat_data in data.get("categories", []):
                    cat = Category(
                        category_id=cat_data["category_id"],
                        name=cat_data["name"],
                        description=cat_data.get("description", ""),
                        parent=cat_data.get("parent"),
                        children=cat_data.get("children", []),
                    )
                    self._categories[cat.category_id] = cat
            except Exception as e:
                logger.warning(f"Failed to load categories: {e}")

    def _snapshot_skills(self) -> dict:
        return {
            "skills": [self._serialize_skill(s) for s in self._skills.values()],
            "updated_at": datetime.now().isoformat(),
        }

    def _snapshot_categories(self) -> dict:
        return {
            "categories": [c.__dict__ for c in self._categories.values()],
        }

    def _persist_state(self, skills_data: dict, categories_data: dict) -> None:
        try:
            skills_file = self.storage_path / "skills.json"
            atomic_write_json(skills_file, skills_data)

            categories_file = self.storage_path / "categories.json"
            atomic_write_json(categories_file, categories_data)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _save_state(self) -> None:
        with self._lock:
            skills_data = self._snapshot_skills()
            categories_data = self._snapshot_categories()
        self._persist_state(skills_data, categories_data)

    def _serialize_skill(self, skill: Skill) -> dict:
        return {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "skill_type": skill.skill_type.value,
            "author": skill.author,
            "category": skill.category,
            "tags": skill.tags,
            "status": skill.status.value,
            "access": skill.access.value,
            "code": skill.code,
            "config": skill.config,
            "parameters": skill.parameters,
            "version": skill.version,
            "versions": [v.to_dict() for v in skill.versions],
            "dependencies": skill.dependencies,
            "rating": skill.rating.to_dict(),
            "rating_feedback": list(skill.rating.feedback)[:self._max_feedback_per_skill],
            "rating_total": skill.rating.total_score,
            "rating_count": skill.rating.count,
            "usage_total": skill.usage.total_uses,
            "usage_last": skill.usage.last_used.isoformat() if skill.usage.last_used else None,
            "usage_users": list(skill.usage.unique_users),
            "created_at": skill.created_at.isoformat(),
            "updated_at": skill.updated_at.isoformat(),
            "published_at": skill.published_at.isoformat() if skill.published_at else None,
            "shared_with": skill.shared_with,
            "metadata": skill.metadata,
        }

    def _deserialize_skill(self, data: dict) -> Skill:
        skill = Skill(
            skill_id=data["skill_id"],
            name=data["name"],
            description=data["description"],
            skill_type=SkillType(data["skill_type"]),
            author=data["author"],
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            status=SkillStatus(data.get("status", "draft")),
            access=AccessLevel(data.get("access", "public")),
            code=data.get("code", ""),
            config=data.get("config", {}),
            parameters=data.get("parameters", {}),
            version=data.get("version", "1.0.0"),
            dependencies=data.get("dependencies", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            published_at=datetime.fromisoformat(data["published_at"]) if data.get("published_at") else None,
            shared_with=data.get("shared_with", []),
            metadata=data.get("metadata", {}),
        )

        skill.rating.total_score = data.get("rating_total", 0)
        skill.rating.count = data.get("rating_count", 0)
        skill.rating.rating = skill.rating.total_score / skill.rating.count if skill.rating.count > 0 else 0
        skill.rating.feedback = deque(data.get("rating_feedback", []), maxlen=self._max_feedback_per_skill)

        skill.usage.total_uses = data.get("usage_total", 0)
        if data.get("usage_last"):
            skill.usage.last_used = datetime.fromisoformat(data["usage_last"])
        skill.usage.unique_users = set(data.get("usage_users", []))

        return skill

    def _index_skill(self, skill: Skill) -> None:
        self._author_skills[skill.author].add(skill.skill_id)
        self._category_index[skill.category].add(skill.skill_id)
        self._type_index[skill.skill_type].add(skill.skill_id)
        for tag in skill.tags:
            self._tag_index[tag.lower()].add(skill.skill_id)

    def _unindex_skill(self, skill: Skill) -> None:
        self._author_skills[skill.author].discard(skill.skill_id)
        self._category_index[skill.category].discard(skill.skill_id)
        self._type_index[skill.skill_type].discard(skill.skill_id)
        for tag in skill.tags:
            self._tag_index[tag.lower()].discard(skill.skill_id)

    def register(
        self,
        name: str,
        description: str,
        skill_type: str | SkillType,
        author: str,
        code: str = "",
        category: str = "general",
        tags: list[str] | None = None,
        dependencies: list[str] | None = None,
        config: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        access: str | AccessLevel = AccessLevel.PUBLIC,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            if len(self._skills) >= self._max_skills:
                self._cleanup_unused()

            skill_id = f"skill_{secrets.token_hex(8)}"

            if isinstance(skill_type, str):
                skill_type = SkillType(skill_type)
            if isinstance(access, str):
                access = AccessLevel(access)

            skill = Skill(
                skill_id=skill_id,
                name=name,
                description=description,
                skill_type=skill_type,
                author=author,
                category=category,
                tags=tags or [],
                code=code,
                config=config or {},
                parameters=parameters or {},
                dependencies=dependencies or [],
                access=access,
                metadata=metadata or {},
            )

            checksum = hashlib.sha256(code.encode()).hexdigest()[:16]
            skill.versions.append(SkillVersion(
                version="1.0.0",
                changes="Initial version",
                checksum=checksum,
            ))

            self._skills[skill_id] = skill
            self._index_skill(skill)

            if category in self._categories:
                self._categories[category].skill_count += 1

            skills_data = self._snapshot_skills()
            categories_data = self._snapshot_categories()

        self._persist_state(skills_data, categories_data)

        logger.info(f"Registered skill: {skill_id} by {author}")
        return skill_id

    def get(self, skill_id: str, user_id: str | None = None, include_code: bool = True) -> Skill | None:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill:
                return None

            if not self._can_access(skill, user_id):
                return None

            return skill

    def update(
        self,
        skill_id: str,
        author: str,
        changes: dict[str, Any],
    ) -> Skill | None:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill:
                return None

            if skill.author != author:
                raise PermissionError("Only the author can update this skill")

            self._unindex_skill(skill)

            for key, value in changes.items():
                if key in ["name", "description", "category", "status", "access", "code", "config", "parameters", "dependencies", "tags"]:
                    setattr(skill, key, value)

            skill.updated_at = datetime.now()

            if "code" in changes:
                checksum = hashlib.sha256(skill.code.encode()).hexdigest()[:16]
                parts = skill.version.split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                new_version = ".".join(parts)

                skill.versions.append(SkillVersion(
                    version=new_version,
                    changes=changes.get("changes_message", "Updated"),
                    checksum=checksum,
                ))
                skill.version = new_version

            self._index_skill(skill)
            skills_data = self._snapshot_skills()
            categories_data = self._snapshot_categories()

        self._persist_state(skills_data, categories_data)

        return skill

    def publish(self, skill_id: str, author: str) -> Skill | None:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill:
                return None

            if skill.author != author:
                raise PermissionError("Only the author can publish this skill")

            skill.status = SkillStatus.PUBLISHED
            skill.published_at = datetime.now()
            skill.updated_at = datetime.now()

            skills_data = self._snapshot_skills()

        self._persist_state(skills_data, {})
        logger.info(f"Skill published: {skill_id}")
        return skill

    def deprecate(self, skill_id: str, author: str, reason: str = "") -> bool:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill:
                return False

            if skill.author != author:
                raise PermissionError("Only the author can deprecate this skill")

            skill.status = SkillStatus.DEPRECATED
            skill.updated_at = datetime.now()
            skill.metadata["deprecation_reason"] = reason

            skills_data = self._snapshot_skills()

        self._persist_state(skills_data, {})
        return True

    def delete(self, skill_id: str, author: str) -> bool:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill:
                return False

            if skill.author != author:
                raise PermissionError("Only the author can delete this skill")

            self._unindex_skill(skill)
            skill.status = SkillStatus.REMOVED

            del self._skills[skill_id]

            if skill.category in self._categories:
                self._categories[skill.category].skill_count -= 1

            skills_data = self._snapshot_skills()
            categories_data = self._snapshot_categories()

        self._persist_state(skills_data, categories_data)
        return True

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        category: str | None = None,
        skill_type: str | SkillType | None = None,
        author: str | None = None,
        status: SkillStatus = SkillStatus.PUBLISHED,
        sort_by: str = "relevance",
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        with self._lock:
            candidates = set(self._skills.keys())

            if tags:
                for tag in tags:
                    tag_matches = self._tag_index.get(tag.lower(), set())
                    candidates = candidates.intersection(tag_matches)

            if category:
                candidates = candidates.intersection(self._category_index.get(category, set()))

            if skill_type:
                if isinstance(skill_type, str):
                    skill_type = SkillType(skill_type)
                candidates = candidates.intersection(self._type_index.get(skill_type, set()))

            if author:
                candidates = candidates.intersection(self._author_skills.get(author, set()))

            results = []
            query_lower = query.lower()

            for skill_id in candidates:
                skill = self._skills[skill_id]

                if skill.status != status and status is not None:
                    continue

                if query:
                    score = 0
                    if query_lower in skill.name.lower():
                        score += 10
                    if query_lower in skill.description.lower():
                        score += 5
                    for tag in skill.tags:
                        if query_lower in tag.lower():
                            score += 3

                    if score == 0:
                        continue
                else:
                    score = skill.rating.rating

                results.append((skill, score))

            if sort_by == "relevance":
                results.sort(key=lambda x: (-x[1], -x[0].rating.rating))
            elif sort_by == "rating":
                results.sort(key=lambda x: (-x[0].rating.rating, -x[0].rating.count))
            elif sort_by == "usage":
                results.sort(key=lambda x: -x[0].usage.total_uses)
            elif sort_by == "recent":
                results.sort(key=lambda x: x[0].updated_at, reverse=True)

            return [skill.to_dict() for skill, _ in results[offset:offset + limit]]

    def rate(
        self,
        skill_id: str,
        rating: int,
        user_id: str = "",
        feedback: str = "",
    ) -> bool:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill:
                return False

            if rating < 1 or rating > 5:
                raise ValueError("Rating must be between 1 and 5")

            skill.rating.add_rating(rating, feedback, user_id)

            skills_data = self._snapshot_skills()

        self._persist_state(skills_data, {})
        return True

    def record_use(self, skill_id: str, user_id: str = "") -> bool:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill:
                return False

            skill.usage.record_use(user_id)
            skills_data = self._snapshot_skills()

        self._persist_state(skills_data, {})
        return True

    def share(self, skill_id: str, author: str, user_ids: list[str]) -> bool:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill:
                return False

            if skill.author != author:
                raise PermissionError("Only the author can share this skill")

            skill.access = AccessLevel.SHARED
            for user_id in user_ids:
                if user_id not in skill.shared_with:
                    skill.shared_with.append(user_id)

            skills_data = self._snapshot_skills()

        self._persist_state(skills_data, {})
        return True

    def _can_access(self, skill: Skill, user_id: str | None) -> bool:
        if skill.access == AccessLevel.PUBLIC:
            return True
        if skill.access == AccessLevel.PRIVATE:
            return user_id == skill.author
        if skill.access == AccessLevel.SHARED:
            return user_id == skill.author or user_id in skill.shared_with
        return False

    def get_categories(self) -> list[dict]:
        return [cat.to_dict() for cat in sorted(self._categories.values(), key=lambda c: c.category_id)]

    def create_category(
        self,
        category_id: str,
        name: str,
        description: str = "",
        parent: str | None = None,
    ) -> Category:
        with self._lock:
            if category_id in self._categories:
                return self._categories[category_id]

            category = Category(
                category_id=category_id,
                name=name,
                description=description,
                parent=parent,
            )

            self._categories[category_id] = category

            if parent and parent in self._categories:
                self._categories[parent].children.append(category_id)

            skills_data = self._snapshot_skills()
            categories_data = self._snapshot_categories()

        self._persist_state(skills_data, categories_data)
        return category

    def get_by_author(self, author: str, include_drafts: bool = False) -> list[dict]:
        with self._lock:
            skill_ids = self._author_skills.get(author, set())
            skills = []
            for sid in skill_ids:
                skill = self._skills.get(sid)
                if skill:
                    if include_drafts or skill.status == SkillStatus.PUBLISHED:
                        skills.append(skill.to_dict())
            return skills

    def get_trending(self, days: int = 7, limit: int = 10) -> list[dict]:
        with self._lock:
            cutoff = datetime.now() - timedelta(days=days)

            trending = []
            for skill in self._skills.values():
                if skill.status != SkillStatus.PUBLISHED:
                    continue

                recent_uses = sum(
                    count for day, count in skill.usage.daily_uses.items()
                    if datetime.strptime(day, "%Y-%m-%d") >= cutoff
                )

                if recent_uses > 0:
                    trending.append((skill, recent_uses))

            trending.sort(key=lambda x: -x[1])
            return [skill.to_dict() for skill, _ in trending[:limit]]

    def get_top_rated(self, min_ratings: int = 5, limit: int = 10) -> list[dict]:
        with self._lock:
            rated = [
                skill for skill in self._skills.values()
                if skill.status == SkillStatus.PUBLISHED
                and skill.rating.count >= min_ratings
            ]
            rated.sort(key=lambda x: (-x.rating.rating, -x.rating.count))
            return [skill.to_dict() for skill in rated[:limit]]

    def get_stats(self) -> dict:
        with self._lock:
            type_counts = defaultdict(int)
            status_counts = defaultdict(int)

            for skill in self._skills.values():
                type_counts[skill.skill_type.value] += 1
                status_counts[skill.status.value] += 1

            return {
                "total_skills": len(self._skills),
                "by_type": dict(type_counts),
                "by_status": dict(status_counts),
                "categories": len(self._categories),
                "authors": len(self._author_skills),
            }

    def _cleanup_unused(self, days: int = 90, keep_min: int = 100) -> int:
        if len(self._skills) <= keep_min:
            return 0

        cutoff = datetime.now() - timedelta(days=days)

        to_remove = []
        for skill_id, skill in self._skills.items():
            if skill.status == SkillStatus.REMOVED:
                continue
            if skill.usage.last_used and skill.usage.last_used < cutoff:
                to_remove.append(skill_id)

        for skill_id in to_remove:
            self.delete(skill_id, self._skills[skill_id].author)

        return len(to_remove)

    def export_skill(self, skill_id: str, author: str) -> dict | None:
        with self._lock:
            skill = self._skills.get(skill_id)
            if not skill or skill.author != author:
                return None

            return self._serialize_skill(skill)

    def import_skill(
        self,
        data: dict,
        author: str,
        overwrite: bool = False,
    ) -> str | None:
        with self._lock:
            existing = self._skills.get(data.get("skill_id"))
            if existing and not overwrite:
                return None

            if existing:
                self._unindex_skill(existing)

            skill = self._deserialize_skill(data)
            skill.author = author
            skill.created_at = datetime.now()
            skill.updated_at = datetime.now()

            if existing:
                skill.skill_id = existing.skill_id
            else:
                skill.skill_id = f"skill_{secrets.token_hex(8)}"

            self._skills[skill.skill_id] = skill
            self._index_skill(skill)

            skills_data = self._snapshot_skills()
            categories_data = self._snapshot_categories()

        self._persist_state(skills_data, categories_data)

        return skill.skill_id


_skills_registry: SkillsRegistry | None = None
_registry_lock = threading.Lock()


def get_skills_registry() -> SkillsRegistry:
    global _skills_registry
    if _skills_registry is None:
        with _registry_lock:
            if _skills_registry is None:
                _skills_registry = SkillsRegistry()
    return _skills_registry


def skills_register(**kwargs) -> str:
    return get_skills_registry().register(**kwargs)


def skills_search(query: str = "", **kwargs) -> list[dict]:
    return get_skills_registry().search(query=query, **kwargs)


def skills_get(skill_id: str, **kwargs) -> dict | None:
    skill = get_skills_registry().get(skill_id, **kwargs)
    return skill.to_dict() if skill else None


def skills_rate(skill_id: str, rating: int, **kwargs) -> bool:
    return get_skills_registry().rate(skill_id, rating, **kwargs)


def skills_trending(limit: int = 10) -> list[dict]:
    return get_skills_registry().get_trending(limit=limit)


def skills_top_rated(limit: int = 10) -> list[dict]:
    return get_skills_registry().get_top_rated(limit=limit)
