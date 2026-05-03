import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Meghdoot Nowcast",
  description: "Live short-range satellite nowcasting over India",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
