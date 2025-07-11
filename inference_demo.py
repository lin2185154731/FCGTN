
import argparse, numpy as np, pandas as pd, torch, torch.nn as nn

class SlimLSTM(nn.Module):
    """Minimal single-layer LSTM -> Linear that outputs
    (mu, log_var) for a 30‑min horizon (600 time‑steps)."""
    def __init__(self, input_dim=5, hidden_dim=16, horizon=600):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head  = nn.Linear(hidden_dim, horizon * 2)  # mu + log_var
        self.horizon = horizon

    def forward(self, x):
        # x: [B, T, C]
        _, (h_n, _) = self.lstm(x)
        h = h_n.squeeze(0)
        out = self.head(h)            # [B, horizon*2]
        mu, log_var = out.chunk(2, dim=-1)
        sigma = (log_var / 2).exp()
        return mu, sigma

def main(csv_file, model_weights, output_json):
    horizon = 600
    df = pd.read_csv(csv_file)
    # simple normalisation
    data = df.values.astype("float32")
    data = (data - data.mean(0)) / (data.std(0) + 1e-6)
    # use the last 1800 samples (~90 min @3s) as context
    context = torch.from_numpy(data[-1800:]).unsqueeze(0)  # [1, T, C]
    model = SlimLSTM(input_dim=data.shape[1], horizon=horizon)
    state_dict = torch.load(model_weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[warning] state_dict keys mismatch -> using random init")
    model.eval()
    with torch.no_grad():
        mu, sigma = model(context)
        mu, sigma = mu.squeeze(0).numpy(), sigma.squeeze(0).numpy()
    # save results
    ts = np.arange(1, horizon + 1) * 3  # seconds ahead
    result = {"seconds_ahead": ts.tolist(),
              "mu": mu.tolist(),
              "sigma": sigma.tolist()}
    with open(output_json, "w") as f:
        import json; json.dump(result, f, indent=2)
    print(f"Saved prediction to {output_json}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="synthetic_data.csv")
    p.add_argument("--weights", default="model.pt")
    p.add_argument("--out", default="forecast.json")
    args = p.parse_args()
    main(args.csv, args.weights, args.out)
