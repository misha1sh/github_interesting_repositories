<script lang="ts">
	import { _ } from 'svelte-i18n';
	import { locale, toggleLocale } from '$lib/i18n/index';
	import RecommendForm, { type FormParams } from '$lib/RecommendForm.svelte';
	import ProgressLog from '$lib/ProgressLog.svelte';
	import ResultsPanel, { type RepoResult, type ClusterInfo } from '$lib/ResultsPanel.svelte';

	type LogEntry = {
		id: number;
		event: string;
		message: string;
		timestamp: string;
	};

	let loading = $state(false);
	let done = $state(false);
	let logEntries = $state<LogEntry[]>([]);
	let results = $state<RepoResult[]>([]);
	let clusters = $state<ClusterInfo[]>([]);
	let errorMessage = $state('');

	let nextId = 0;

	function pushLog(event: string, message: string) {
		const now = new Date();
		const ts = now.toTimeString().slice(0, 8);
		logEntries = [...logEntries, { id: nextId++, event, message, timestamp: ts }];
	}

	async function handleSubmit(params: FormParams) {
		loading = true;
		done = false;
		logEntries = [];
		results = [];
		clusters = [];
		errorMessage = '';

		pushLog('progress', $_('log.startingFor', { values: { username: params.username } }));

		try {
			const res = await fetch('/api/recommend', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(params),
			});

			if (!res.ok) {
				const body = await res.json().catch(() => ({}));
				const msgs = body.errors?.join(', ') ?? `HTTP ${res.status}`;
				errorMessage = msgs;
				pushLog('error', $_('log.requestFailed', { values: { msg: msgs } }));
				loading = false;
				return;
			}

			const reader = res.body?.getReader();
			if (!reader) {
				errorMessage = $_('log.noStream');
				loading = false;
				return;
			}

			const decoder = new TextDecoder();
			let buffer = '';

			while (true) {
				const { done: streamDone, value } = await reader.read();
				if (streamDone) break;

				buffer += decoder.decode(value, { stream: true });
				const blocks = buffer.split('\n\n');
				buffer = blocks.pop() ?? '';

				for (const block of blocks) {
					if (!block.trim()) continue;
					let eventType = 'progress';
					let dataStr = '';
					for (const line of block.split('\n')) {
						if (line.startsWith('event:')) eventType = line.slice(6).trim();
						else if (line.startsWith('data:')) dataStr = line.slice(5).trim();
					}
					if (!dataStr) continue;

					let data: Record<string, unknown>;
					try {
						data = JSON.parse(dataStr);
					} catch {
						continue;
					}

					const msg = (data.message as string) ?? '';

					if (eventType === 'result') {
						results = [
							...results,
							{
								rank: data.rank as number,
								name: data.name as string,
								score: data.score as number,
								calibrated: (data.calibrated as number | null) ?? null,
								pct_rank: (data.pct_rank as number | null) ?? null,
								matching_tags: (data.matching_tags as string[]) ?? [],
								cluster_id: (data.cluster_id as number | null) ?? null,
								source: (data.source as 'cluster' | 'overall') ?? 'overall',
							},
						];
						pushLog(eventType, msg);
					} else if (eventType === 'cluster') {
						clusters = [
							...clusters,
							{
								cluster_id: data.cluster_id as number,
								cluster_total: data.cluster_total as number,
								member_count: data.member_count as number,
								top_tags: (data.top_tags as string[]) ?? [],
								member_names: (data.member_names as string[]) ?? [],
							},
						];
						pushLog(eventType, msg);
					} else if (eventType === 'done') {
						const success = data.success as boolean;
						done = true;
						loading = false;
						if (!success && data.error) {
							errorMessage = data.error as string;
							pushLog('error', $_('log.doneWithError', { values: { error: data.error as string } }));
						} else {
							pushLog('done', $_('log.complete', { values: { count: results.length } }));
						}
					} else if (eventType === 'error') {
						pushLog('error', msg);
					} else {
						pushLog('progress', msg);
					}
				}
			}
		} catch (err) {
			errorMessage = String(err);
			pushLog('error', $_('log.connectionError', { values: { error: String(err) } }));
		} finally {
			loading = false;
		}
	}

	let currentLocale = $derived($locale?.slice(0, 2) ?? 'en');
</script>

<svelte:head>
	<title>{$_('header.title')}</title>
</svelte:head>

<!-- App Bar -->
<header class="sticky top-0 z-50 backdrop-blur border-b border-surface-700" style="background: linear-gradient(135deg, #0f1923 0%, #111827 50%, #0c1a2e 100%);">
	<div class="max-w-screen-xl mx-auto px-4 h-14 flex items-center gap-3">
		<svg xmlns="http://www.w3.org/2000/svg" class="size-6 shrink-0" style="filter: drop-shadow(0 0 6px rgba(56,189,248,0.5)); color: #38bdf8;" viewBox="0 0 24 24" fill="currentColor">
			<path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.866-.013-1.7-2.782.603-3.369-1.342-3.369-1.342-.454-1.154-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.091-.646.35-1.087.636-1.337-2.22-.252-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0 1 12 6.836a9.58 9.58 0 0 1 2.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.202 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.744 0 .267.18.579.688.481C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
		</svg>
		<h1 class="text-lg font-bold tracking-tight" style="background: linear-gradient(to right, #ffffff 0%, #7dd3fc 55%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; filter: drop-shadow(0 0 10px rgba(56,189,248,0.3));">{$_('header.title')}</h1>

		<!-- Language switcher -->
		<button
			type="button"
			class="ml-auto flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border border-surface-600 hover:border-primary-500 hover:text-primary-400 text-surface-300 transition-colors"
			onclick={toggleLocale}
			title="Switch language"
		>
			<svg xmlns="http://www.w3.org/2000/svg" class="size-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
			</svg>
			{currentLocale === 'ru' ? 'RU' : 'EN'}
		</button>

		{#if loading}
			<span class="flex items-center gap-2 text-sm text-primary-400">
				<span class="size-2 rounded-full bg-primary-400 animate-ping"></span>
				{$_('header.running')}
			</span>
		{:else if done}
			<span class="text-sm text-success-400">
				{$_('header.done', { values: { count: results.length } })}
			</span>
		{/if}
	</div>
</header>

<main class="max-w-screen-xl mx-auto px-4 py-6">
	<div class="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6">

		<!-- Left: Form -->
		<aside class="flex flex-col gap-4">
			<div class="card preset-tonal-surface p-5">
				<h2 class="text-base font-semibold mb-4 flex items-center gap-2">
					<svg xmlns="http://www.w3.org/2000/svg" class="size-4 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
					</svg>
					{$_('sections.parameters')}
				</h2>
				<RecommendForm {loading} onSubmit={handleSubmit} />
			</div>

			{#if errorMessage}
				<div class="card preset-tonal-error p-4">
					<p class="text-error-400 text-sm font-semibold mb-1">{$_('error.label')}</p>
					<p class="text-error-300 text-sm break-words">{errorMessage}</p>
				</div>
			{/if}
		</aside>

		<!-- Right: Progress + Results -->
		<div class="flex flex-col gap-5">

			<!-- Progress Log -->
			<section>
				<h2 class="text-base font-semibold mb-2 flex items-center gap-2">
					<svg xmlns="http://www.w3.org/2000/svg" class="size-4 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
					</svg>
					{$_('sections.liveProgress')}
					{#if logEntries.length > 0}
						<span class="badge preset-tonal-surface text-xs px-2">{logEntries.length}</span>
					{/if}
				</h2>
				<ProgressLog entries={logEntries} {loading} {done} />
			</section>

			<!-- Results -->
			<section>
				<h2 class="text-base font-semibold mb-2 flex items-center gap-2">
					<svg xmlns="http://www.w3.org/2000/svg" class="size-4 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
					</svg>
					{$_('sections.recommendations')}
				</h2>
				<ResultsPanel {results} {clusters} {loading} {done} />
			</section>
		</div>
	</div>
</main>
