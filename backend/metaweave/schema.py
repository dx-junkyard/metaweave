"""Pydantic schemas for extracted problem structures."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ProblemStatement(BaseModel):
    """Background context and the core problem to be solved."""

    background: str = Field(default="", description="Background context of the research")
    problem: str = Field(default="", description="Core problem the paper addresses")


class Hypothesis(BaseModel):
    """Research hypothesis or conjecture."""

    statement: str = Field(default="", description="Main hypothesis")
    rationale: str = Field(default="", description="Rationale behind the hypothesis")


class Methodology(BaseModel):
    """Approach and methods used in the research."""

    approach: str = Field(default="", description="High-level approach")
    techniques: list[str] = Field(default_factory=list, description="Specific techniques or tools used")


class Constraints(BaseModel):
    """Constraints, assumptions, and limitations."""

    assumptions: list[str] = Field(default_factory=list, description="Underlying assumptions")
    limitations: list[str] = Field(default_factory=list, description="Known limitations")


class CausalEdge(BaseModel):
    """A directed edge in the causal/relational graph."""

    source: str = Field(description="Source variable")
    target: str = Field(description="Target variable")
    relation: str = Field(default="causes", description="Type of relation (causes, inhibits, correlates, ...)")


class AbstractStructure(BaseModel):
    """Abstract structure extracted from the paper: variables and causal edges."""

    variables: list[str] = Field(default_factory=list, description="Extracted variables / key concepts")
    edges: list[CausalEdge] = Field(default_factory=list, description="Causal or relational edges")


class PaperStructure(BaseModel):
    """Full extracted structure for a single paper."""

    paper_id: str = Field(description="Unique identifier (e.g. arXiv ID)")
    title: str = Field(default="")
    problem: ProblemStatement = Field(default_factory=ProblemStatement)
    hypothesis: Hypothesis = Field(default_factory=Hypothesis)
    methodology: Methodology = Field(default_factory=Methodology)
    constraints: Constraints = Field(default_factory=Constraints)
    abstract_structure: AbstractStructure = Field(default_factory=AbstractStructure)
    review_status: ReviewStatus = Field(default=ReviewStatus.PENDING)
    reviewer_notes: str = Field(default="")
