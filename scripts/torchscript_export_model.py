import torch
from signalp6.model_for_deployment import EnsembleBertCRFModel

model_list = [
    "test_0_val_1",
    "test_0_val_2",
    "test_1_val_0",
    "test_1_val_2",
    "test_2_val_0",
    "test_2_val_1",
]
base_path = "/work3/felteu/tagging_checkpoints/signalp_6/"

dummy_input_ids = torch.ones(1, 73, dtype=int)
dummy_input_mask = torch.ones(1, 73, dtype=int)

model = EnsembleBertCRFModel([base_path + x for x in model_list], None)
model.eval()
model = torch.jit.trace(model, (dummy_input_ids, dummy_input_mask))
model.save("/checkpoints/ensemble_scripted.pt")
