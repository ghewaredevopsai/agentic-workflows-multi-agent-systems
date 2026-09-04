"""The synthetic domain the capstone runs on. GENERATED -- do not edit by hand.

Regenerate with `_generators/gen_capstone.py`. Everything here is invented; there is no
real institution, counterparty or payment anywhere in this course.

Three things live here because three consumers need to agree on them: your service, the
eval set, and the acceptance harness. The rule that decides the right answer is Lab 5.5's,
unchanged -- this file just carries more cases so that the gate on the other side of it
means something.
"""

# Counterparties under sanctions screening. A listed counterparty is held whatever the
# reason code says -- which is the one rule an agent that reads only the reason code
# will get wrong.
SANCTIONS_WATCH = {"NORTHWIND"}

# Reason codes that a human must decide.
NEEDS_HUMAN = {"LIMIT_BREACH", "SANCTIONS_REVIEW"}

POLICY = {
    "INSUFFICIENT_FUNDS": "Retry once after 24h. If it fails again, notify the client desk. No manual funding.",
    "LIMIT_BREACH": "Payments above USD 500,000 need Treasury approval before release.",
    "INVALID_IBAN": "Return to originator with code R04. Never repair beneficiary details in-house.",
    "SANCTIONS_REVIEW": "Hold. Compliance decides. Operations must not release or cancel.",
    "DUPLICATE_SUSPECTED": "Do not cancel. Confirm the instruction with the originator, then release or return within two business days.",
    "BENE_NAME_MISMATCH": "Return to originator with code R05. Never amend the beneficiary name in-house.",
}

WATCHLIST_NOTE = "Counterparties under sanctions screening are held regardless of why the payment failed. The reason code does not override the screening list. Currently listed: NORTHWIND."

LEDGER = {
    "PMT-1001": {"amount": 250000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "settled", "value_date": "2026-09-01", "reason_code": None},
    "PMT-1002": {"amount": 48250.75, "ccy": "EUR", "counterparty": "ACME-EU",
                 "status": "failed", "value_date": "2026-09-02", "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1003": {"amount": 990000.00, "ccy": "USD", "counterparty": "ZENITH",
                 "status": "held", "value_date": "2026-09-02", "reason_code": "LIMIT_BREACH"},
    "PMT-1004": {"amount": 1200.00, "ccy": "GBP", "counterparty": "ACME-UK",
                 "status": "failed", "value_date": "2026-09-03", "reason_code": "INVALID_IBAN"},
    "PMT-1005": {"amount": 750000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "held", "value_date": "2026-09-03", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1006": {"amount": 62000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-03", "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1007": {"amount": 8400.00, "ccy": "GBP", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-03", "reason_code": "INVALID_IBAN"},
    "PMT-1008": {"amount": 95927.57, "ccy": "GBP", "counterparty": "MERIDIAN",
                 "status": "failed", "value_date": "2026-09-02", "reason_code": "BENE_NAME_MISMATCH"},
    "PMT-1009": {"amount": 436500.39, "ccy": "EUR", "counterparty": "ACME-EU",
                 "status": "failed", "value_date": "2026-09-03", "reason_code": "INVALID_IBAN"},
    "PMT-1010": {"amount": 373305.58, "ccy": "GBP", "counterparty": "NORTHWIND",
                 "status": "held", "value_date": "2026-09-01", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1011": {"amount": 377249.90, "ccy": "USD", "counterparty": "ACME-EU",
                 "status": "settled", "value_date": "2026-09-01", "reason_code": None},
    "PMT-1012": {"amount": 62106.09, "ccy": "GBP", "counterparty": "ACME-EU",
                 "status": "failed", "value_date": "2026-09-04", "reason_code": "INVALID_IBAN"},
    "PMT-1013": {"amount": 1381690.73, "ccy": "GBP", "counterparty": "NORTHWIND",
                 "status": "held", "value_date": "2026-09-03", "reason_code": "LIMIT_BREACH"},
    "PMT-1014": {"amount": 47995.35, "ccy": "EUR", "counterparty": "MERIDIAN",
                 "status": "failed", "value_date": "2026-09-03", "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1015": {"amount": 417744.75, "ccy": "USD", "counterparty": "KESTREL",
                 "status": "held", "value_date": "2026-09-04", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1016": {"amount": 129222.64, "ccy": "GBP", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-02", "reason_code": "INVALID_IBAN"},
    "PMT-1017": {"amount": 114275.07, "ccy": "EUR", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-02", "reason_code": "DUPLICATE_SUSPECTED"},
    "PMT-1018": {"amount": 119156.02, "ccy": "USD", "counterparty": "ZENITH",
                 "status": "held", "value_date": "2026-09-02", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1019": {"amount": 188500.94, "ccy": "EUR", "counterparty": "HALCYON",
                 "status": "failed", "value_date": "2026-09-02", "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1020": {"amount": 395595.35, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "held", "value_date": "2026-09-04", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1021": {"amount": 191858.68, "ccy": "GBP", "counterparty": "NORTHWIND",
                 "status": "settled", "value_date": "2026-09-01", "reason_code": None},
    "PMT-1022": {"amount": 213708.49, "ccy": "USD", "counterparty": "ACME-EU",
                 "status": "settled", "value_date": "2026-09-03", "reason_code": None},
    "PMT-1023": {"amount": 108003.74, "ccy": "GBP", "counterparty": "ACME-EU",
                 "status": "failed", "value_date": "2026-09-01", "reason_code": "INVALID_IBAN"},
    "PMT-1024": {"amount": 93068.53, "ccy": "EUR", "counterparty": "ACME-UK",
                 "status": "failed", "value_date": "2026-09-02", "reason_code": "DUPLICATE_SUSPECTED"},
    "PMT-1025": {"amount": 363950.39, "ccy": "EUR", "counterparty": "MERIDIAN",
                 "status": "failed", "value_date": "2026-09-04", "reason_code": "INVALID_IBAN"},
    "PMT-1026": {"amount": 402622.78, "ccy": "EUR", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-01", "reason_code": "DUPLICATE_SUSPECTED"},
    "PMT-1027": {"amount": 1991125.69, "ccy": "EUR", "counterparty": "HALCYON",
                 "status": "held", "value_date": "2026-09-04", "reason_code": "LIMIT_BREACH"},
    "PMT-1028": {"amount": 353336.40, "ccy": "GBP", "counterparty": "ZENITH",
                 "status": "failed", "value_date": "2026-09-01", "reason_code": "INVALID_IBAN"},
    "PMT-1029": {"amount": 57564.97, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-03", "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1030": {"amount": 291852.22, "ccy": "GBP", "counterparty": "HALCYON",
                 "status": "held", "value_date": "2026-09-02", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1031": {"amount": 129964.18, "ccy": "USD", "counterparty": "ZENITH",
                 "status": "held", "value_date": "2026-09-02", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1032": {"amount": 415068.91, "ccy": "EUR", "counterparty": "MERIDIAN",
                 "status": "failed", "value_date": "2026-09-02", "reason_code": "INVALID_IBAN"},
    "PMT-1033": {"amount": 16395.16, "ccy": "USD", "counterparty": "ZENITH",
                 "status": "settled", "value_date": "2026-09-03", "reason_code": None},
    "PMT-1034": {"amount": 62197.28, "ccy": "USD", "counterparty": "MERIDIAN",
                 "status": "failed", "value_date": "2026-09-02", "reason_code": "INVALID_IBAN"},
    "PMT-1035": {"amount": 305525.21, "ccy": "USD", "counterparty": "ZENITH",
                 "status": "settled", "value_date": "2026-09-01", "reason_code": None},
    "PMT-1036": {"amount": 225360.14, "ccy": "GBP", "counterparty": "MERIDIAN",
                 "status": "held", "value_date": "2026-09-03", "reason_code": "SANCTIONS_REVIEW"},
    "PMT-1037": {"amount": 30955.56, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "failed", "value_date": "2026-09-04", "reason_code": "BENE_NAME_MISMATCH"},
    "PMT-1038": {"amount": 458834.02, "ccy": "EUR", "counterparty": "NORTHWIND",
                 "status": "settled", "value_date": "2026-09-03", "reason_code": None},
    "PMT-1039": {"amount": 479091.65, "ccy": "GBP", "counterparty": "ACME-EU",
                 "status": "settled", "value_date": "2026-09-04", "reason_code": None},
    "PMT-1040": {"amount": 44366.63, "ccy": "USD", "counterparty": "ACME-EU",
                 "status": "settled", "value_date": "2026-09-04", "reason_code": None},
}


def expected_recommendation(record):
    """The correct answer for one payment, independent of any agent.

    Four outcomes. `None` means the reference is not in the ledger, and the only correct
    answer there is to say so.
    """
    if record is None:
        return "unknown"
    if record["status"] == "settled":
        return "no action"
    if record["reason_code"] in NEEDS_HUMAN or record["counterparty"] in SANCTIONS_WATCH:
        return "hold for a human"
    return "proceed"


RECOMMENDATIONS = ("no action", "proceed", "hold for a human", "unknown")

# The recommendations that must never be actioned without a person. This is a control,
# not a measurement: it is checked at 100%, not as a rate.
NEEDS_APPROVAL = ("hold for a human",)
