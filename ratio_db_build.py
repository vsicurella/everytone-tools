
from ratio_db import *
from farey import farey
from basis import farey_set_to_basis, get_triplet_basis

def BuildHarmonics(db: RatioDb, limit=1024):
    if limit < 1:
        print("Skipping harmonics")
        return

    num_harmonics = db.get_count("harmonics")
    for i in range(num_harmonics + 1, limit + 1):
        db.add_harmonic(i)
    print("finished with harmonics up to " + str(limit))

def BuildDyads(db:RatioDb, dyadLimit=300, harmonic_extension=64):
    if dyadLimit < 1:
        print("Skipping dyads")
        return
    num_dyads = db.get_count("dyads")
    int_limit_set = farey(dyadLimit)
    periodic_set = farey_set_to_basis(int_limit_set, harmonic_extension)
    i = 0
    for d,n in periodic_set:
        if n == 0 or d == 0:
            continue
        if num_dyads > 0:
            dyad = db.find_dyad(n, d)
            if dyad is not None:
                continue
        db.add_dyad(n, d)
        i += 1
    print(f"Saved {i} dyads up to int limit {dyadLimit} up to harmonic {harmonic_extension}")

def BuildTriads(db:RatioDb, integerLimit=100, complexity_limit=27_000_000, harmonic_extension=64, triad_start_harmonic=1):
    num_triads = db.get_count("triads")
    print(f"Currently {num_triads} triads")

    print(f'BuildTriads - generating triad set with params {integerLimit} - {complexity_limit} - {harmonic_extension} - {triad_start_harmonic}...')
    triads = get_triplet_basis(integerLimit, harmonic_extension, complexity_limit, start_harmonic=triad_start_harmonic)
    print(f'BuildTriads - generated {len(triads)} triads')

    i = 0
    last_root = 0
    for r,m,d in triads:
        if r != last_root:
            last_root = r
            print(f"BuildTriads - on root {r}")

        if r > m or r > d or m > d:
            raise Exception(f"Invalid triad syntax: {r}:{m}:{d}")
        if num_triads > 0:
            triad = db.find_triad(r, m, d)
            if triad is not None:
                continue
        db.add_triad(r, m, d)
        i += 1

    print(f"BuildTriads - Saved {i} triads with int limit {integerLimit} up to harmonic {harmonic_extension} with complexity limit {complexity_limit}")

def BuildRatioDb(db: RatioDb, harmLimit=1024, dyadLimit=512, triadLimit=128, harmonic_extension=32, triad_start_harmonic=1):
    BuildHarmonics(db, harmLimit)
    BuildDyads(db, dyadLimit, harmonic_extension)
    BuildTriads(db, triadLimit, harmonic_extension=harmonic_extension, triad_start_harmonic=triad_start_harmonic)

if __name__ == "__main__":
    ratioDb = RatioDb()
    # BuildRatioDb(ratioDb, 0, 0, 128, 4, 100)
