from typing import Dict, List, Optional

from . import constants as c
from .gaussian_scorer import GaussianScorer


class GaussianExpansionScorer(GaussianScorer):
  """Gaussian convolution scorer for expansion population (coverage model).

  This scorer inherits all Gaussian scoring logic but filters ratings to the
  same population as MFExpansionScorer (coreGroups | expansionGroups, plus
  unassigned raters). It uses MFExpansionScorer prescoring outputs for rater
  factors and runs as a final-scoring coverage model after the MF expansion
  model, parallel to how GaussianScorer covers MFCoreScorer.
  """

  def __init__(
    self,
    seed: Optional[int] = None,
    threads: int = c.defaultNumThreads,
    saveIntermediateState: bool = False,
  ) -> None:
    """Configure GaussianExpansionScorer object.

    Args:
      seed: if not None, seed value to ensure deterministic execution
      threads: number of threads to use for intra-op parallelism in pytorch
      saveIntermediateState: if True, save intermediate state for debugging
    """
    super().__init__(
      # Mirror MFExpansionScorer population: core + expansion groups.
      includedGroups=(c.coreGroups | c.expansionGroups),
      # MFExpansionScorer does not exclude topics (mf_base default).
      excludeTopics=False,
      includeUnassigned=True,
      captureThreshold=0.5,
      seed=seed,
      threads=threads,
      saveIntermediateState=saveIntermediateState,
    )

  def get_prescoring_name(self):
    return "MFExpansionScorer"

  def get_name(self):
    return "GaussianExpansionScorer"

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: c.gaussianExpansionNoteInterceptKey,
      c.internalNoteFactor1Key: c.gaussianExpansionNoteFactor1Key,
      c.internalActiveRulesKey: c.gaussianExpansionActiveRulesKey,
      c.numFinalRoundRatingsKey: c.gaussianExpansionNumFinalRoundRatingsKey,
      c.internalNoteInterceptNoHighVolKey: c.gaussianExpansionNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: c.gaussianExpansionNoteInterceptNoCorrelatedKey,
      c.internalNoteInterceptPopulationSampledKey: c.gaussianExpansionNoteInterceptPopulationSampledKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalRatingStatusKey: c.gaussianExpansionRatingStatusKey,
    }

  def _get_user_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default user column names to custom names for a specific model."""
    return {}

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      c.gaussianExpansionNoteInterceptKey,
      c.gaussianExpansionNoteFactor1Key,
      c.gaussianExpansionRatingStatusKey,
      c.gaussianExpansionActiveRulesKey,
      c.gaussianExpansionNumFinalRoundRatingsKey,
      c.gaussianExpansionNoteInterceptNoHighVolKey,
      c.gaussianExpansionNoteInterceptNoCorrelatedKey,
      c.gaussianExpansionNoteInterceptPopulationSampledKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return [
      c.raterParticipantIdKey,
    ]

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return [
      c.noteIdKey,
    ]
