import json, sys
inv = json.load(open(sys.argv[1]))
for g, r in inv['gen'].items():
    print(f"gen/{g:18s} npz {r['n_npz']:5d} csv {r['n_csv']:5d} links {r['n_links']} broken {len(r['broken'])} "
          f"npz-no-csv {len(r['npz_without_csv'])} csv-no-npz {len(r['csv_without_npz'])} expected {r.get('expected')} "
          f"missing {r.get('missing_expected', [])[:5]} extra {len(r.get('extra', []))}:{r.get('extra', [])[:6]} events {r['events']}")
for s, r in inv['stage'].items():
    if r.get('csv_link_dirs_not_mirrored'):
        print('NOT MIRRORED', s, r['csv_link_dirs_not_mirrored'])
    if 'meta_events' in r:
        print(f"{s:30s} {r['meta_events']}")
