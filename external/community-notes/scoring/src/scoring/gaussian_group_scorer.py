from typing import Dict, List, Optional, Set, Tuple

from . import constants as c
from .gaussian_scorer import GaussianScorer

import pandas as pd


# Modeling group scored by GaussianCoreGroupScorer (also in c.coreGroups).
# Applied via ApplyNMRGroupModelResult to demote CRH/NYH when this model disagrees.
coreGroupScoringGroup = 21


class GaussianGroupScorer(GaussianScorer):
  def __init__(
    self,
    includedGroups: Set[int],
    groupId: int,
    strictInclusion: bool = False,
    seed: Optional[int] = None,
    groupThreshold: float = 0.8,
    saveIntermediateState: bool = False,
    minMeanNoteScore: float = 0.05,
    crhThreshold: float = 0.40,
    crnhThresholdIntercept: float = -0.05,
    crnhThresholdNoteFactorMultiplier: float = -0.8,
    crnhThresholdNMIntercept: float = -0.15,
    crnhThresholdUCBIntercept: float = -0.5,
    enableRatioCrnh: bool = True,
    crhSuperThreshold: Optional[float] = None,
    crhThresholdNoHighVol: float = 0.37,
    crhThresholdNoCorrelated: float = 0.37,
    lowDiligenceThreshold: float = 0.263,
    factorThreshold: float = 0.5,
    firmRejectThreshold: Optional[float] = None,
    largeFactorRequiresCrh: bool = True,
    tagFilterPercentile: int = 95,
    incorrectFilterThreshold: float = 2.5,
    threads: int = 4,
    crhParams: c.GaussianParams = c.gaussianCrhParams,
    crnhParams: c.GaussianParams = c.gaussianCrnhParams,
    calculateBins: bool = False,
    centeredBins: bool = False,
  ) -> None:
    """Configure GaussianGroupScorer object.

    Notice that each GaussianGroupScorer defines column names by appending the groupId to
    column prefixes which are constant.  Dynamically defining the column names allows the
    group scorer to be instantiated multiple times while maintaining the property that
    the columns attached by each scorer remain unique.  Once all scorers have ran, we
    validate that each note was scored by at most one group scorer and then coalesce
    all of the group scoring columns and remove the groupId suffix.
    """
    super().__init__(
      includedGroups=includedGroups,
      strictInclusion=strictInclusion,
      includeUnassigned=False,
      captureThreshold=groupThreshold,
      seed=seed,
      saveIntermediateState=saveIntermediateState,
      threads=threads,
      minMeanNoteScore=minMeanNoteScore,
      crhThreshold=crhThreshold,
      crnhThresholdIntercept=crnhThresholdIntercept,
      crnhThresholdNoteFactorMultiplier=crnhThresholdNoteFactorMultiplier,
      crnhThresholdNMIntercept=crnhThresholdNMIntercept,
      crnhThresholdUCBIntercept=crnhThresholdUCBIntercept,
      enableRatioCrnh=enableRatioCrnh,
      crhSuperThreshold=crhSuperThreshold,
      crhThresholdNoHighVol=crhThresholdNoHighVol,
      crhThresholdNoCorrelated=crhThresholdNoCorrelated,
      lowDiligenceThreshold=lowDiligenceThreshold,
      factorThreshold=factorThreshold,
      firmRejectThreshold=firmRejectThreshold,
      largeFactorRequiresCrh=largeFactorRequiresCrh,
      tagFilterPercentile=tagFilterPercentile,
      incorrectFilterThreshold=incorrectFilterThreshold,
      crhParams=crhParams,
      crnhParams=crnhParams,
      calculateBins=calculateBins,
      centeredBins=centeredBins,
      # Override GaussianScorer default (True). MFGroupScorer does not exclude topics.
      excludeTopics=False,
    )
    assert groupId > 0, "groupId must be positive.  0 is reserved for unassigned."
    self._groupId = groupId
    self._groupNoteInterceptKey = f"{c.groupNoteInterceptKey}_{self._groupId}"
    self._groupNoteFactor1Key = f"{c.groupNoteFactor1Key}_{self._groupId}"
    self._groupRatingStatusKey = f"{c.groupRatingStatusKey}_{self._groupId}"
    self._groupInternalActiveRulesKey = f"{c.groupInternalActiveRulesKey}_{self._groupId}"
    self._groupNumFinalRoundRatingsKey = f"{c.groupNumFinalRoundRatingsKey}_{self._groupId}"
    self._modelingGroupKey = f"{c.modelingGroupKey}_{self._groupId}"
    self._groupNoteInterceptNoHighVolKey = f"{c.groupNoteInterceptNoHighVolKey}_{self._groupId}"
    self._groupNoteInterceptNoCorrelatedKey = (
      f"{c.groupNoteInterceptNoCorrelatedKey}_{self._groupId}"
    )

  def get_prescoring_name(self):
    return f"MFGroupScorer_{self._groupId}"

  def get_name(self):
    return f"GaussianGroupScorer_{self._groupId}"

  def _prescore_notes_and_users(
    self,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollmentRaw: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, c.PrescoringMetaScorerOutput]:
    raise Exception("GaussianGroupScorer should not be used for prescoring")

  def _get_note_col_mapping(self) -> Dict[str, str]:
    """Returns a dict mapping default note column names to custom names for a specific model."""
    return {
      c.internalNoteInterceptKey: self._groupNoteInterceptKey,
      c.internalNoteFactor1Key: self._groupNoteFactor1Key,
      c.internalRatingStatusKey: self._groupRatingStatusKey,
      c.internalActiveRulesKey: self._groupInternalActiveRulesKey,
      c.numFinalRoundRatingsKey: self._groupNumFinalRoundRatingsKey,
      c.lowDiligenceNoteInterceptKey: c.lowDiligenceLegacyNoteInterceptKey,
      c.internalNoteInterceptNoHighVolKey: self._groupNoteInterceptNoHighVolKey,
      c.internalNoteInterceptNoCorrelatedKey: self._groupNoteInterceptNoCorrelatedKey,
    }

  def get_scored_notes_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the scoredNotes output."""
    return [
      c.noteIdKey,
      self._groupNoteInterceptKey,
      self._groupNoteFactor1Key,
      self._groupRatingStatusKey,
      self._groupInternalActiveRulesKey,
      self._modelingGroupKey,
      self._groupNumFinalRoundRatingsKey,
      self._groupNoteInterceptNoHighVolKey,
      self._groupNoteInterceptNoCorrelatedKey,
    ]

  def get_helpfulness_scores_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the helpfulnessScores output."""
    return []

  def get_auxiliary_note_info_cols(self) -> List[str]:
    """Returns a list of columns which should be present in the auxiliaryNoteInfo output."""
    return []

  def _get_dropped_user_cols(self) -> List[str]:
    return super()._get_dropped_user_cols() + [
      c.raterParticipantIdKey,
    ]

  def _postprocess_output(
    self,
    noteScores: pd.DataFrame,
    userScores: pd.DataFrame,
    ratings: pd.DataFrame,
    noteStatusHistory: pd.DataFrame,
    userEnrollment: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    noteScores, userScores = super()._postprocess_output(
      noteScores, userScores, ratings, noteStatusHistory, userEnrollment
    )
    userScores = userScores.merge(
      userEnrollment[[c.participantIdKey, c.modelingGroupKey]].rename(
        columns={c.participantIdKey: c.raterParticipantIdKey}
      ),
      how="left",
    )
    userScores = userScores[userScores[c.modelingGroupKey].isin(self._includedGroups)]
    userScores = userScores.drop(columns=c.modelingGroupKey)
    noteScores[self._modelingGroupKey] = self._groupId
    return noteScores, userScores


class GaussianCoreGroupScorer(GaussianGroupScorer):
  """Gaussian group scorer for coreGroupScoringGroup (21).

  Uses MFCoreScorer prescoring (group 21 is in coreGroups; there is no dedicated MF
  group model). Writes standard group* columns that coalesce with other group models
  and is applied via ApplyNMRGroupModelResult.

  Large-factor notes (|factor| > factorThreshold) are FIRM_REJECT even without prior CRH,
  so ApplyNMRGroupModelResult can demote final CRH when this model disagrees.
  """

  def __init__(
    self,
    seed: Optional[int] = None,
    threads: int = 4,
    saveIntermediateState: bool = False,
    groupThreshold: float = 0.75,
    crhThreshold: float = 0.56,
    factorThreshold: float = 0.5,
    crhSuperThreshold: Optional[float] = None,
  ) -> None:
    super().__init__(
      includedGroups={coreGroupScoringGroup},
      groupId=coreGroupScoringGroup,
      groupThreshold=groupThreshold,
      seed=seed,
      threads=threads,
      saveIntermediateState=saveIntermediateState,
      crhThreshold=crhThreshold,
      factorThreshold=factorThreshold,
      # FR large-factor notes even if they never hit CRH in this scorer (not CRH-only).
      largeFactorRequiresCrh=False,
      crhSuperThreshold=crhSuperThreshold,
      # Stronger opposite-side prior (priorSlope=1.2 vs default 0.5) and lower clip/minPrior
      # floors so polarized notes can score lower on the other side of the spectrum.
      crhParams=c.GaussianParams(clipLower=0.03, minPrior=0.1, priorSlope=1.2),
    )

  def get_prescoring_name(self):
    # These raters are scored in MF core prescoring; no dedicated group MF model.
    return "MFCoreScorer"
