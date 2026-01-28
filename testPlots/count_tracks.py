from podio.reading import get_reader

# change this to your file
INPUT = "/ceph/omunkombwe/fit__idea_o1_v03_60.root"

def count_tracks_per_event(path):
    reader = get_reader(path)
    events = reader.get("events")
    for iev, event in enumerate(events):
        tracks = event.get("CDCHTracks")
        print(f"event {iev}: {tracks.size()} tracks")

if __name__ == "__main__":
    count_tracks_per_event(INPUT)
