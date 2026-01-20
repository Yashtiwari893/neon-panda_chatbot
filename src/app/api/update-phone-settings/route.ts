import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabaseClient";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { phone_number, intent, system_prompt, auth_token, origin } = body;

    if (!phone_number) {
      return NextResponse.json(
        { error: "Phone number is required" },
        { status: 400 }
      );
    }

    console.log("Updating phone settings for:", phone_number);

    // 🔹 Check existing mappings
    const { data: existingMappings, error: fetchError } = await supabase
      .from("phone_document_mapping")
      .select("*")
      .eq("phone_number", phone_number);

    if (fetchError) {
      throw fetchError;
    }

    const updateData: any = {};
    if (intent !== undefined) updateData.intent = intent;
    if (system_prompt !== undefined) updateData.system_prompt = system_prompt;
    if (auth_token !== undefined) updateData.auth_token = auth_token;
    if (origin !== undefined) updateData.origin = origin;

    // 🔹 CASE 1: Mapping exists → UPDATE
    if (existingMappings && existingMappings.length > 0) {
      const { error: updateError } = await supabase
        .from("phone_document_mapping")
        .update(updateData)
        .eq("phone_number", phone_number);

      if (updateError) throw updateError;

      // 🔹 Also sync credentials to files
      if (auth_token || origin) {
        const fileIds = existingMappings
          .map(m => m.file_id)
          .filter(Boolean);

        if (fileIds.length > 0) {
          const fileUpdate: any = {};
          if (auth_token) fileUpdate.auth_token = auth_token;
          if (origin) fileUpdate.origin = origin;

          await supabase
            .from("rag_files")
            .update(fileUpdate)
            .in("id", fileIds);
        }
      }

      return NextResponse.json({
        success: true,
        message: "Phone settings updated",
        mode: "updated",
      });
    }

    // 🔹 CASE 2: No mapping exists → CREATE (UPSERT)
    const { error: insertError } = await supabase
      .from("phone_document_mapping")
      .insert({
        phone_number,
        intent: intent ?? null,
        system_prompt: system_prompt ?? null,
        auth_token: auth_token ?? null,
        origin: origin ?? null,
        file_id: null,
      });

    if (insertError) throw insertError;

    return NextResponse.json({
      success: true,
      message: "Phone settings created",
      mode: "created",
    });

  } catch (error) {
    console.error("Update phone settings error:", error);

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to update phone settings",
      },
      { status: 500 }
    );
  }
}
