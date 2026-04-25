<script lang="ts">
	import { tick } from 'svelte';
	import { _ } from 'svelte-i18n';

	type LogEntry = {
		id: number;
		event: string;
		message: string;
		timestamp: string;
	};

	type Props = {
		entries: LogEntry[];
		loading: boolean;
		done: boolean;
	};

	let { entries, loading, done }: Props = $props();

	const STAGES = [
		{ id: 'fetch',   labelKey: 'progress.stages.fetch',   patterns: ['Fetching starred', 'Fetched page', 'Fetching owned', 'Found', 'Added'] },
		{ id: 'enrich',  labelKey: 'progress.stages.enrich',  patterns: ['Enriching missing', 'LLM enrichment', 'cache hit', 'to generate'] },
		{ id: 'embed',   labelKey: 'progress.stages.embed',   patterns: ['Computing all repo embed', 'Moving model to', 'Preloading', 'Loading trained'] },
		{ id: 'profile', labelKey: 'progress.stages.profile', patterns: ['Building user embedding', 'Building user tag', 'found in graph', 'starred repo'] },
		{ id: 'cluster', labelKey: 'progress.stages.cluster', patterns: ['Clustering starred', 'Auto-selected', 'Cluster ', '[Cluster'] },
		{ id: 'score',   labelKey: 'progress.stages.score',   patterns: ['Top ', 'Recommendations for', 'Overall', 'Score spread', 'Computing overall'] },
		{ id: 'done',    labelKey: 'progress.stages.done',    patterns: [] },
	];

	let currentStageIdx = $state(-1);
	let latestMessage = $state('');
	let showLog = $state(false);
	let scrollEl = $state<HTMLDivElement | undefined>(undefined);

	$effect(() => {
		const last = entries[entries.length - 1];
		if (!last) return;

		const msg = last.message;
		latestMessage = msg.trim();

		for (let i = STAGES.length - 2; i >= 0; i--) {
			if (STAGES[i].patterns.some(p => msg.includes(p))) {
				if (i > currentStageIdx) currentStageIdx = i;
				break;
			}
		}

		tick().then(() => {
			if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
		});
	});

	$effect(() => {
		if (done && currentStageIdx < STAGES.length - 1) {
			currentStageIdx = STAGES.length - 1;
		}
	});

	function stageState(idx: number): 'done' | 'active' | 'pending' {
		if (done) return 'done';
		if (idx < currentStageIdx) return 'done';
		if (idx === currentStageIdx) return 'active';
		return 'pending';
	}

	let progressPct = $derived(
		currentStageIdx < 0
			? 0
			: Math.round(((currentStageIdx + (done ? 1 : 0.5)) / STAGES.length) * 100)
	);

	function eventColor(event: string): string {
		switch (event) {
			case 'error':   return 'text-error-400';
			case 'result':  return 'text-success-400';
			case 'cluster': return 'text-warning-400';
			case 'done':    return 'text-primary-400 font-bold';
			default:        return 'text-surface-300';
		}
	}
</script>

<div class="flex flex-col gap-3">
	{#if entries.length === 0 && !loading}
		<div class="card preset-tonal-surface p-4 text-sm text-surface-500 italic text-center">
			{$_('progress.empty')}
		</div>
	{:else}
		<!-- Progress bar -->
		<div class="card preset-tonal-surface p-4 flex flex-col gap-3">
			<div class="flex items-center justify-between text-xs text-surface-400">
				<span class="font-mono">{progressPct}%</span>
				{#if loading}
					<span class="flex items-center gap-1.5 text-primary-400">
						<span class="size-1.5 rounded-full bg-primary-400 animate-ping inline-block"></span>
						{$_('progress.running')}
					</span>
				{:else if done}
					<span class="text-success-400">{$_('progress.complete')}</span>
				{/if}
			</div>

			<!-- Bar -->
			<div class="h-2 bg-surface-700 rounded-full overflow-hidden">
				<div
					class="h-full rounded-full transition-all duration-500 {done ? 'bg-success-500' : 'bg-primary-500'}"
					style="width: {progressPct}%"
				></div>
			</div>

			<!-- Stages -->
			<div class="flex flex-col gap-1.5 mt-1">
				{#each STAGES as stage, i}
					{@const state = stageState(i)}
					<div class="flex items-center gap-2 text-xs">
						<span class="shrink-0 size-4 flex items-center justify-center">
							{#if state === 'done'}
								<svg class="size-4 text-success-400" viewBox="0 0 20 20" fill="currentColor">
									<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clip-rule="evenodd"/>
								</svg>
							{:else if state === 'active'}
								<span class="size-3 rounded-full border-2 border-primary-400 border-t-transparent animate-spin inline-block"></span>
							{:else}
								<span class="size-2.5 rounded-full border border-surface-600 inline-block"></span>
							{/if}
						</span>
						<span class="{state === 'done' ? 'text-success-400' : state === 'active' ? 'text-primary-300 font-semibold' : 'text-surface-600'}">
							{$_(stage.labelKey)}
						</span>
					</div>
				{/each}
			</div>

			<!-- Latest message -->
			{#if latestMessage && (loading || done)}
				<p class="text-xs font-mono text-surface-400 truncate border-t border-surface-700 pt-2 mt-1">
					› {latestMessage}
				</p>
			{/if}
		</div>

		<!-- Collapsible log -->
		<div>
			<button
				type="button"
				class="flex items-center gap-2 text-xs text-surface-500 hover:text-surface-300 transition-colors w-full"
				onclick={() => (showLog = !showLog)}
			>
				<svg
					class="size-3 transition-transform {showLog ? 'rotate-90' : ''}"
					viewBox="0 0 20 20" fill="currentColor"
				>
					<path fill-rule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clip-rule="evenodd"/>
				</svg>
				{showLog
					? $_('progress.hideLog', { values: { count: entries.length } })
					: $_('progress.showLog', { values: { count: entries.length } })}
			</button>

			{#if showLog}
				<div
					bind:this={scrollEl}
					class="mt-2 overflow-y-auto bg-surface-950 rounded-lg p-3 font-mono text-xs leading-relaxed border border-surface-800"
					style="max-height: 320px;"
				>
					{#each entries as entry (entry.id)}
						<div class="flex gap-2 hover:bg-surface-900 px-1 rounded">
							<span class="text-surface-600 select-none shrink-0 w-16 text-right">{entry.timestamp}</span>
							<span class="{eventColor(entry.event)} break-all whitespace-pre-wrap">{entry.message}</span>
						</div>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
</div>
