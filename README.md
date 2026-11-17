# GTA VI launch-day Twitch forecast

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`: forecast how
many people watch Grand Theft Auto VI live on Twitch in each hour of its
release day (19 November 2026, UTC), as five quantiles per hour, from the
first hour's count alone.

## Method

`main.py` (Python 3.13, standard library only, no network, deterministic)
fits, at run time and from the case file's own comparables, one OLS line in
log space per hour:

```
log viewers(h) = a_h + b_h * log viewers(1) + c_h * log viewers(-1h)
```

Training launches are the rolling-midnight family only: past launches that
were already being watched in hour 1 (hour-1 viewers >= 2% of their
release-day peak) and first crossed 10% of that peak within the 24 hours
before their release day started. That excludes daytime simultaneous
launches (hour 1 near zero) and early-access launches (their release day
is not their launch day). The second feature, viewers in the hour before
the day started, measures how strong the already-unlocked eastern markets
are. Quantiles are the fitted mean plus normal quantiles times 0.9 * the
hour's residual standard deviation.

## Local validation

Practice answers re-read from the kit's own documented source (SullyGnome
hourly averages; hour-1 prints match the case files exactly):

| model | mean pinball loss (log scale) |
|---|---|
| house baseline starter | 0.387 |
| this entry | **0.163** |

Per practice launch: Wolverine 0.199, Red Dead Redemption 2 0.137,
God of War Ragnarök 0.122, Spider-Man 2 0.197. Leave-one-out over the
comparable launches in the same family: 0.251 (same fit without the
launch-family filter: 0.263). The official practice score is the server's
grade on submission; the figures above are local reproductions.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` for each.
One case runs in about 0.1 s.

## License

MIT — see `LICENSE`.
