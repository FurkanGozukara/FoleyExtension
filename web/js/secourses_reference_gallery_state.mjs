/**
 * Resolve the prompt during a ComfyUI node configure pass.
 *
 * The serialized widget value is authoritative only for the first workflow
 * hydration. After the editor is live, its value must survive attachment-only
 * reconfiguration, even when ComfyUI supplies a stale widget snapshot.
 */
export function resolveConfiguredPrompt(currentPrompt, widgetPrompt, hasHydratedPrompt, promptTouched) {
    const hydrateFromWidget = !hasHydratedPrompt && !promptTouched;
    return {
        value: hydrateFromWidget ? (widgetPrompt ?? "") : currentPrompt,
        hydrateFromWidget,
    };
}
