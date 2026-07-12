import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "CoRAG — Multi-hop answers over your documents",
  description:
    "Upload your documents and ask complex, multi-hop questions. CoRAG retrieves iteratively and answers with citations.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
