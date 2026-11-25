import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { answer, deleteThread, getThread, listThreads } from "@/lib/chat";
import { defaultModelId, llmEnabled } from "@/lib/judge/provider";
import { StoreError } from "@/lib/store";
import { activeWorkspace, runInWorkspace } from "@/lib/workspaces";
import { inWorkspace, errorResponse, jsonBody } from "@/lib/http";
export const runtime = "nodejs";
const threadId = z.string().uuid();

/** The conversations of the workspace, or one of them with its messages (`?thread=<id>`). */
export async function GET(request: NextRequest) {
  try {
    return inWorkspace(request, () => {
      const thread = request.nextUrl.searchParams.get("thread");
      return NextResponse.json(
        thread
          ? getThread(threadId.parse(thread))
          : {
              threads: listThreads(),
              configured: llmEnabled(),
              defaultModel: defaultModelId(),
            },
      );
    });
  } catch (e) {
    return errorResponse(e);
  }
}

/** One question to the workspace chat, answered as server-sent events (`data: <json>`, then
 *  `data: [DONE]`):
 *    {"type":"thread","thread":{…}}        the conversation (new when no threadId was sent)
 *    {"type":"step","step":{tool,input}}   a lookup the model made
 *    {"type":"sources","sources":[…]}      every numbered source of this answer so far
 *    {"type":"text","delta":"…"}           the answer as it is written
 *    {"type":"message","message":{…}}      the stored answer
 *    {"type":"error","detail":"…"}
 *  This spends money on the configured OpenRouter key and sends workspace text to OpenRouter.
 *  A client that disconnects stops the model call; the answer is then not stored. */
export async function POST(request: NextRequest) {
  try {
    return await inWorkspace(request, async () => {
      const body = await jsonBody(request);
      if (!llmEnabled())
        throw new StoreError(
          "The chat needs an OpenRouter key. Add one on the Evaluators page (Traces → Evaluators).",
          503,
        );
      const workspaceId = activeWorkspace().id;
      const encoder = new TextEncoder();
      const abort = new AbortController();
      let open = true;
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          const send = (frame: string) => {
            if (!open) return;
            try {
              controller.enqueue(encoder.encode(`data: ${frame}\n\n`));
            } catch {
              open = false; // the client went away
              abort.abort();
            }
          };
          void runInWorkspace(workspaceId, () =>
            answer(body, (event) => send(JSON.stringify(event)), abort.signal),
          )
            .catch((error: Error) =>
              send(
                JSON.stringify({
                  type: "error",
                  detail:
                    error instanceof z.ZodError
                      ? "The message is empty or too long."
                      : error.message,
                }),
              ),
            )
            .finally(() => {
              send("[DONE]");
              if (open) controller.close();
              open = false;
            });
        },
        cancel() {
          open = false;
          abort.abort();
        },
      });
      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache, no-transform",
          "X-Accel-Buffering": "no",
        },
      });
    });
  } catch (e) {
    return errorResponse(e);
  }
}

/** Remove one conversation and its messages (`?thread=<id>`). Records are not touched. */
export async function DELETE(request: NextRequest) {
  try {
    return inWorkspace(request, () =>
      NextResponse.json(
        deleteThread(
          threadId.parse(request.nextUrl.searchParams.get("thread")),
        ),
      ),
    );
  } catch (e) {
    return errorResponse(e);
  }
}
