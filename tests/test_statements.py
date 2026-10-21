from finrag.statements import page_kinds, question_kinds, route

BALANCE = ("Table of Contents NIKE, INC. CONSOLIDATED BALANCE SHEETS MAY 31, (In millions) 2021 2020 "
           "ASSETS Current assets: Cash and equivalents $ 9,889 $ 8,348 Short-term investments 3,587 "
           "439 Accounts receivable, net 4,463 2,749 Inventories 6,854 7,367 Total current assets "
           "26,291 20,556")
INCOME = ("Walmart Inc. Consolidated Statements of Income Fiscal Years Ended January 31, (Amounts in "
          "millions) 2019 2018 2017 Revenues: Net sales $ 510,329 $ 495,761 $ 481,317 Membership "
          "and other income 4,076 4,582 4,556 Total revenues 514,405 500,343 485,873")
CASH = ("SQUARE, INC. CONSOLIDATED STATEMENTS OF CASH FLOWS (In thousands) Year Ended December 31, "
        "2020 2019 2018 Net income (loss) $ 213,105 $ 375,446 $ (38,453) Depreciation 84,212 75,598 "
        "60,961 Non-cash interest 76,129 33,478 28,512")


def test_page_kinds_finds_statements_and_skips_pages_that_only_mention_them():
    assert page_kinds(BALANCE) == {"balance"}
    assert page_kinds(INCOME) == {"income"}
    assert page_kinds(CASH) == {"cashflow"}
    assert page_kinds("Report of Independent Registered Public Accounting Firm. We have audited "
                      "the consolidated balance sheets 2022 2021 1 2 3 4 5 6 7 8") == set()
    assert page_kinds("NOTES TO CONSOLIDATED FINANCIAL STATEMENTS balance sheets 1 2 3 4 5") == set()
    prose = "Consolidated Balance Sheets discussion of liquidity and capital resources " * 5
    assert page_kinds(prose) == set()  # names the statement but isn't a table


def test_question_kinds_named_and_implied():
    assert question_kinds("Using the statement of financial position, what is ...") == {"balance"}
    assert question_kinds("What is FY2018 capex according to the cash flow statement?") == {"cashflow"}
    assert question_kinds("Does Paypal have positive working capital?") == {"balance"}
    assert question_kinds("What drove the effective tax rate change?") == {"income"}
    assert question_kinds("Who is the new CEO?") == set()
    assert question_kinds("working capital", implied=False) == set()


def test_route_moves_requested_kinds_first_and_keeps_their_order():
    kinds = [frozenset(), frozenset({"income"}), frozenset(), frozenset({"balance"}),
             frozenset({"income", "balance"})]
    assert route([0, 1, 2, 3, 4], kinds, frozenset({"balance"})) == [3, 4, 0, 1, 2]
    assert route([2, 1, 0], kinds, frozenset()) == [2, 1, 0]
