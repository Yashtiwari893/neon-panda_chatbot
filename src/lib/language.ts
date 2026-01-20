import Groq from "groq-sdk";

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY!,
});

/**
 * Detect language of a given text
 * Always returns a safe lowercase string
 */
export async function detectLanguage(text: string): Promise<string> {
    try {
        const completion = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            temperature: 0,
            messages: [
                {
                    role: "system",
                    content:
                        "Detect the language of the given text. Reply with ONLY the language name.",
                },
                {
                    role: "user",
                    content: text,
                },
            ],
        });

        const language =
            completion.choices?.[0]?.message?.content?.toLowerCase();

        return language || "english";
    } catch (error) {
        console.error("Language detection failed:", error);
        return "english";
    }
}
