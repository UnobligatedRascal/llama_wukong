<script lang="ts">
	import { parseWriteFileMeta } from './parsers/write-file';
	import ToolCallBlock from './ToolCallBlock.svelte';
	import { XCircle } from '@lucide/svelte';
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import { MAX_HEIGHT_CODE_BLOCK, RESULT_STAT_SEPARATOR } from '$lib/constants';
	import { FileTypeText } from '$lib/enums';
	import { toolsStore } from '$lib/stores';
	import type { AgenticSection } from '$lib/types';
	import { abbreviateHome, formatJsonPretty } from '$lib/utils';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { isStreaming, onToggle, open, section }: Props = $props();

	const writeFileMeta = $derived(parseWriteFileMeta(section));
	const home = $derived(toolsStore.serverHome);
</script>

<ToolCallBlock {isStreaming} meta={writeFileMeta} {onToggle} {open} {section}>
	{#snippet titleSnippet()}
		<span class="flex min-w-0 flex-wrap items-baseline gap-x-1">
			<span class="shrink-0 text-muted-foreground">Write file</span>

			<span class="flex min-w-0 items-baseline gap-1.5">
				<span class="min-w-0 overflow-x-auto font-mono" title={writeFileMeta?.filePath}>
					{abbreviateHome(writeFileMeta?.filePath ?? '', home)}
				</span>

				{#if writeFileMeta?.errorMessage}
					<span class="shrink-0 text-xs italic text-muted-foreground/70">(failed)</span>
				{/if}
			</span>
		</span>
	{/snippet}

	{#snippet children(_meta, ctx)}
		{#if writeFileBody?.errorMessage}
			<div
				class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
			>
				<XCircle class="mt-0.5 h-3 w-3 shrink-0" />

				<span>{writeFileBody.errorMessage}</span>
			</div>
		{:else if writeFileBody?.content !== undefined}
			<SyntaxHighlightedCode
<<<<<<< HEAD
				code={meta.content}
				language={meta.language}
=======
				code={writeFileBody.content}
				language={writeFileBody.language}
>>>>>>> ggml-org/master
				maxHeight={MAX_HEIGHT_CODE_BLOCK}
				streaming={ctx.isCodeStreaming}
			/>

			<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
				{#if writeFileBody.resultMessage}
					{writeFileBody.resultMessage}{writeFileBody.bytesWritten != null
						? RESULT_STAT_SEPARATOR
						: ''}{/if}

				{#if writeFileBody.bytesWritten != null}
					<span class="font-mono">{writeFileBody.bytesWritten}</span>
					bytes
				{/if}
			</div>
		{:else if section.toolArgs}
			<!-- no content arg: not streamed yet, or a same-named tool with other arg names -->
			<SyntaxHighlightedCode
				code={formatJsonPretty(section.toolArgs)}
				language={FileTypeText.JSON}
				maxHeight={MAX_HEIGHT_CODE_BLOCK}
				streaming={ctx.isCodeStreaming}
			/>
		{:else if ctx.isStreaming}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Receiving arguments...
			</div>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No content</div>
		{/if}
	{/snippet}
</ToolCallBlock>
