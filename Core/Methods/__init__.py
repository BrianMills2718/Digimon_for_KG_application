"""Pre-defined retrieval method plans.

Each function returns an ExecutionPlan that replaces a Query class.
The PipelineExecutor can run these plans using the typed operator system.
"""

from Core.Methods.basic_local import basic_local_plan
from Core.Methods.basic_global import basic_global_plan
from Core.Methods.lightrag import lightrag_plan
from Core.Methods.fastgraphrag import fastgraphrag_plan
from Core.Methods.hipporag import hipporag_plan
from Core.Methods.tog import tog_plan
from Core.Methods.gr import gr_plan
from Core.Methods.dalk import dalk_plan
from Core.Methods.kgp import kgp_plan
from Core.Methods.med import med_plan

METHOD_PLANS = {
    "basic_local": basic_local_plan,
    "basic_global": basic_global_plan,
    "lightrag": lightrag_plan,
    "fastgraphrag": fastgraphrag_plan,
    "hipporag": hipporag_plan,
    "tog": tog_plan,
    "gr": gr_plan,
    "dalk": dalk_plan,
    "kgp": kgp_plan,
    "med": med_plan,
}
