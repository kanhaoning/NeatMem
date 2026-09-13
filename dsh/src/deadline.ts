/**
 * Shared hard deadline for DSH foreground memory work, adapted from the MemOS
 * dsh adapter's deadline.ts. An optional recall may never hold DSH's prompt
 * path beyond the configured budget.
 */
export async function waitForDeadline<T>(
  operation: Promise<T>,
  options: {
    deadlineAt: number;
    signal: AbortSignal;
    now?: () => number;
    timeoutMessage: string;
  },
): Promise<T> {
  const now = options.now ?? (() => Date.now());
  if (options.signal.aborted) {
    void operation.catch(() => undefined);
    throw options.signal.reason ?? new DOMException("DSH operation aborted", "AbortError");
  }

  let timer: ReturnType<typeof setTimeout> | undefined;
  let onAbort: (() => void) | undefined;
  const cutoff = new Promise<never>((_resolve, reject) => {
    const finish = (error: unknown): void => {
      if (timer !== undefined) clearTimeout(timer);
      options.signal.removeEventListener("abort", onAbort!);
      reject(error);
    };
    onAbort = () =>
      finish(options.signal.reason ?? new DOMException("DSH operation aborted", "AbortError"));
    options.signal.addEventListener("abort", onAbort, { once: true });
    timer = setTimeout(
      () => finish(new DOMException(options.timeoutMessage, "TimeoutError")),
      Math.max(0, options.deadlineAt - now()),
    );
  });

  try {
    // Promise.race installs rejection handlers on both branches, so a late
    // rejection remains observed after the host has already failed open.
    return await Promise.race([operation, cutoff]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    if (onAbort) options.signal.removeEventListener("abort", onAbort);
  }
}

export function isDeadlineTimeout(error: unknown): boolean {
  return error instanceof DOMException && error.name === "TimeoutError";
}
