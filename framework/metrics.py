
from framework.config import JUDGE_MODEL, CONTECTUAL_PRECISION_THRESHOLD, CONTECTUAL_RECALL_THRESHOLD,CONTEXTUAL_RELEVANCY_THRESHOLD
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric

def create_contextual_relevancy_metric():
    return ContextualRelevancyMetric(
        threshold=ContextualRelevancyMetric,
        model = JUDGE_MODEL, 
        include_reason= True
    )

def create_contextual_precision_metric():
    return ContextualPrecisionMetric(
        threshold = CONTECTUAL_PRECISION_THRESHOLD,
        model = JUDGE_MODEL,
        include_reason = True
    )

def create_contextual_recall_metric():
    return ContextualRecallMetric(
        threshold = CONTECTUAL_RECALL_THRESHOLD,
        model = JUDGE_MODEL,
        include_reason = True
    )
