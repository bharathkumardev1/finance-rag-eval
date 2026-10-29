# Judge labels

Status: graded by the author against the FinanceBench reference answers, with AI-assisted evidence gathering. Not blind to AI input.

## How the 60 labels were made

`finrag label-judge` sampled 60 judged (question, answer) pairs from the 400 generated answers, with the judge's verdict hidden. Identical answers from two settings appear once. The sample covers 51 distinct questions, so agreement intervals resample questions rather than rows.

The author graded each pair using the same rubric the judge was given (`RUBRIC.txt`): an answer is correct if it gives the same figure or conclusion as the reference, within rounding or 1%. The author is responsible for every final verdict.

AI tools were used to help find the supporting filings, pull the figures, and check calculations, so each decision could be defended. The labels are therefore research-assisted. They are not independent of AI input.

Verdicts follow the reference, not the filing. Where the reference itself looks wrong or ambiguous, the row is graded against the reference and listed below.

## Rows where the reference or the rubric is debatable

| Row | Question | Note |
|---|---|---|
| 4 | Coca-Cola ROA | Reference 0.01 is a ratio, the answer is 1.42%. Equal after two-decimal rounding of the ratio. |
| 10 | JPMorgan liquidation per share | Reference $66.56 is tangible book value per share (Q1 2021 10-Q). Book value is $82.31. |
| 13 | AMD FY22 revenue drivers | Reference lists EPYC, semi-custom and Xilinx embedded. The answer covers embedded, client and gaming, not EPYC. |
| 27 | Boeing gross margin | Answer says 4.9% for FY2021, filing gives 4.84%. Conclusion matches. |
| 29 | Pfizer spin-off | Reference says Upjohn was being spun off, but it closed on 16 November 2020. Graded incorrect against the reference. The model's "No" is historically right. |
| 37 | Corning working capital | Answer $2,278m, reference $831m. Both say positive. Graded correct, see below. |
| 46 | 3M capital intensity | Judgmental call. Graded against the reference's "No". |

## Row 37: Corning working capital

The question asks whether Corning had positive working capital in FY2022. The reference says yes, $831 million, and notes that it counts only the operating current assets and current liabilities shown on the balance sheet. The answer says yes, $2,278 million.

Both figures come from the FY2022 balance sheet (10-K, December 31, 2022, in millions):

- The answer is total current assets of 7,453 minus total current liabilities of 5,175, which is 2,278. This is the usual definition.
- The reference leaves out cash (1,671) and the current portion of debt (224): receivables 1,721 + inventories 2,904 + other current assets 1,157 - accounts payable 1,804 - other accrued liabilities 3,147 = 831.

The rubric accepts the same figure or the same conclusion. The conclusion matches and the figure is correct under the usual definition, so I graded it correct. The judge marked it incorrect because the figure differs from the reference. This is the one disagreement in the 60 labels. A grader who required the reference's figure would count it incorrect and agreement would be 60 of 60.

## Result

Judge against these labels: see `results/generation.json` (`judge_check`).
