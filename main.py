import yaml
from InSITE_Engine.ingestion.ingest import run_ingestion
from InSITE_Engine.cleaning.clean import basic_clean
from InSITE_Engine.normalisation.normalise import normalise
from InSITE_Engine.features.features import compute_features
from InSITE_Engine.events.events import detect_events
from InSITE_Engine.states.states import derive_states

def main():
    with open('InSITE_Engine/config/config.yaml') as f:
        cfg = yaml.safe_load(f)

    data = run_ingestion(cfg)

    for key, dfs in data.items():
        cleaned = [basic_clean(df) for df in dfs]
        normalised = [normalise(df) for df in cleaned]
        featured = [compute_features(df) for df in normalised]
        events = [detect_events(df) for df in featured]
        states = [derive_states(ev) for ev in events]

    print('InSITE Engine completed successfully.')

if __name__ == '__main__':
    main()
