# Monetization and audience metrics

**Status checked: 10 October 2026.** Official eligibility is determined by YouTube Studio and the policies for the channel's country; this document is a tracking guide, not an eligibility guarantee.

## YouTube Partner Program thresholds

YouTube currently describes two YPP entry levels for eligible regions:

| Level | Subscriber / activity requirements | Shorts route | Watch-hours route |
|---|---|---:|---:|
| Expanded YPP (fan funding and selected Shopping features) | 500 subscribers and 3 valid public uploads in the prior 90 days | 3 million qualified Shorts views in 90 days | 3,000 qualified watch hours in 12 months |
| Standard YPP (ad-revenue sharing) through 31 January 2027 | 1,000 subscribers | 10 million qualified public Shorts views in 90 days | 4,000 qualified watch hours in 12 months |
| Standard YPP for new applicants from 1 February 2027 | 1,000 subscribers | 20 million qualified Shorts views in 90 days | 8,000 qualified watch hours in 12 months |

The expanded-tier requirements are not part of the announced February 2027 change. Other requirements still apply, including country availability, eligible public content, account security, and policy compliance. Confirm current eligibility in YouTube Studio rather than relying on this repo's estimate.

**Shorts progress must use engaged/qualified views, not the newer public `views` count.** Since 31 March 2025, a public Shorts view can count when a Short starts or replays. YouTube says YPP eligibility and Shorts ad-revenue sharing continue to use engaged Shorts views. AutoTube keeps raw public views for reporting and queries YouTube Analytics `engagedViews` for its rolling-window progress proxy. That API value is only complete when the summary reports 100% coverage; the Studio eligibility panel remains authoritative.

## What AutoTube measures

- **`averageViewPercentage`** — average percentage of the video watched per playback. This is AutoTube's *average-view target* metric. It is not the percentage of unique viewers who completed the Short.
- **`audienceWatchRatio`** — relative watch activity at portions of the video, stored as `audience_watch_ratio_curve`. It helps locate relative changes along a video. Replays can make a value exceed 1.0; the final point is **not** a viewer-completion rate.
- **`engagedViews`** — the Shorts-view measure used for the YPP progress proxy when Analytics returns it. Raw `views` remains a separate reporting value.
- **Internal average-view target: 70%.** This is an AutoTube goal for `averageViewPercentage`, not an official YouTube threshold, published distribution gate, or promise of reach. Each video's result must be measured from channel analytics after it has had time to mature.
- **Length and format tests are hypotheses.** The configured 15–20 second duration is an internal experiment for this channel. There is no verified official 25–40 second “dead zone,” universal winning length bracket, or loop-length sweet spot in this repo's evidence.

Older history stored `audienceWatchRatio` curve values under names such as `completion` and `hook_retention`. Those fields have been renamed to watch-ratio telemetry and are no longer used as completion percentages or reward inputs. Rewards are rebuilt against the corrected metrics. A stale dashboard/history snapshot should not be treated as a viewer-completion study.

## Monetization-safe editorial standard

Eligibility is not just a view-count problem. AutoTube is intended to make original, advertiser-friendly Archive 13 Shorts in the `history_mystery` and `science_nature` lanes. Every upload needs a current trend/viral signal and an original angle; a Wikipedia article can support facts but does not by itself prove that a subject is trending. Do not copy or lightly repackage another creator's Short. A repeatable format is acceptable; repeated substance or low-value, mass-produced variations are not.

Reviewers and automated checks should prioritize factual accuracy, meaningful commentary/value-add, distinct scripts and visuals, and advertiser-friendly treatment. YouTube's policy describes categories of inauthentic or reused content, but AutoTube does **not** claim unverified numerical triggers such as a fixed script-variation percentage, daily upload count, commentary share, or retention cutoff. Realistic altered or synthetic content must be disclosed when YouTube's disclosure rules require it; disclosure is not a substitute for originality or policy compliance.

## Sources

- [YouTube Partner Program overview and eligibility](https://support.google.com/youtube/answer/13429240)
- [YPP eligibility threshold changes](https://support.google.com/youtube/answer/12843009)
- [How Shorts views are counted](https://support.google.com/youtube/answer/10059070)
- [YouTube Analytics API metrics](https://developers.google.com/youtube/analytics/metrics)
- [YouTube Analytics API revision history](https://developers.google.com/youtube/analytics/revision_history)
- [YouTube channel monetization policies](https://support.google.com/youtube/answer/1311392)
