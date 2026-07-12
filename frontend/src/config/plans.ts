// Display catalog for the pricing/billing pages. Quotas must stay in sync
// with services/api/src/corag_cloud/billing/plans.py (backend enforcement).

export interface PlanCard {
  id: "trial" | "starter" | "pro" | "team";
  name: string;
  priceEur: number | null;
  questions: number;
  documents: number;
  storage: string;
  blurb: string;
}

export const PLANS: PlanCard[] = [
  {
    id: "trial",
    name: "Trial",
    priceEur: null,
    questions: 25,
    documents: 10,
    storage: "20 MB",
    blurb: "14 days, no card required.",
  },
  {
    id: "starter",
    name: "Starter",
    priceEur: 9,
    questions: 200,
    documents: 50,
    storage: "200 MB",
    blurb: "For personal research corpora.",
  },
  {
    id: "pro",
    name: "Pro",
    priceEur: 29,
    questions: 1000,
    documents: 250,
    storage: "1 GB",
    blurb: "For professionals asking daily.",
  },
  {
    id: "team",
    name: "Team",
    priceEur: 79,
    questions: 5000,
    documents: 1000,
    storage: "5 GB",
    blurb: "For teams with shared knowledge.",
  },
];
