<script lang="ts">
	import { _ } from 'svelte-i18n';

	export type RepoResult = {
		rank: number;
		name: string;
		score: number;
		calibrated: number | null;
		pct_rank: number | null;
		matching_tags: string[];
		cluster_id: number | null;
		source: 'cluster' | 'overall';
	};

	export type ClusterInfo = {
		cluster_id: number;
		cluster_total: number;
		member_count: number;
		top_tags: string[];
		member_names: string[];
	};

	type Props = {
		results: RepoResult[];
		clusters: ClusterInfo[];
		loading: boolean;
		done: boolean;
	};

	let { results, clusters, loading, done }: Props = $props();

	let activeTab = $state<'recommended' | 'clusters'>('recommended');

	const CLUSTER_SCORE_MIN = 0.9;
	const TOP_RECOMMENDED = 10;

	function scoreBar(score: number): number {
		return Math.min(100, Math.max(0, score * 100));
	}

	function scoreColor(score: number): string {
		if (score >= 0.97) return 'bg-success-500';
		if (score >= 0.93) return 'bg-primary-500';
		if (score >= 0.90) return 'bg-warning-500';
		return 'bg-error-500';
	}

	function pctLabel(pct: number | null): string {
		if (pct === null) return '';
		const p = Math.round((1 - pct) * 100);
		return `top ${p}%`;
	}

	let topRecommended = $derived.by(() => {
		const seen = new Set<string>();
		const merged: RepoResult[] = [];
		for (const r of results) {
			if (!seen.has(r.name)) {
				seen.add(r.name);
				merged.push(r);
			}
		}
		return merged
			.sort((a, b) => b.score - a.score)
			.slice(0, TOP_RECOMMENDED);
	});

	let clusterGroups = $derived.by(() => {
		const allClusterResults = results.filter(r => r.cluster_id !== null);
		const map = new Map<number, RepoResult[]>();
		for (const r of allClusterResults) {
			const cid = r.cluster_id!;
			if (!map.has(cid)) map.set(cid, []);
			if (!map.get(cid)!.some(x => x.name === r.name)) {
				map.get(cid)!.push(r);
			}
		}
		for (const c of clusters) {
			if (!map.has(c.cluster_id)) map.set(c.cluster_id, []);
		}
		const order = [...map.keys()].sort((a, b) => a - b);
		return order.map(cid => {
			const allRepos = (map.get(cid) ?? []).sort((a, b) => b.score - a.score);
			return {
				cluster: clusters.find(c => c.cluster_id === cid) ?? null,
				repos: allRepos.filter(r => r.score >= CLUSTER_SCORE_MIN),
				hiddenCount: allRepos.filter(r => r.score < CLUSTER_SCORE_MIN).length,
			};
		});
	});

	let hasAnyResults = $derived(results.length > 0);
</script>

<div class="flex flex-col gap-4">
	{#if !hasAnyResults && !loading}
		<div class="card preset-tonal-surface p-10 text-center">
			<svg xmlns="http://www.w3.org/2000/svg" class="size-12 mx-auto mb-3 text-surface-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z" />
			</svg>
			{#if done}
				<p class="text-surface-400 text-sm">{$_('results.empty.noResults')}</p>
			{:else}
				<p class="text-surface-400 text-sm">{$_('results.empty.waiting')}</p>
			{/if}
		</div>
	{:else}
		<!-- Tabs -->
		<div class="flex gap-1 p-1 bg-surface-800 rounded-xl">
			<button
				type="button"
				class="flex-1 py-2 px-3 rounded-lg text-sm font-semibold transition-all {activeTab === 'recommended' ? 'bg-primary-500 text-white shadow' : 'text-surface-400 hover:text-surface-200'}"
				onclick={() => (activeTab = 'recommended')}
			>
				{$_('results.tabs.recommended')}
				{#if topRecommended.length > 0}
					<span class="ml-1.5 text-xs opacity-75">({topRecommended.length})</span>
				{/if}
			</button>
			<button
				type="button"
				class="flex-1 py-2 px-3 rounded-lg text-sm font-semibold transition-all {activeTab === 'clusters' ? 'bg-primary-500 text-white shadow' : 'text-surface-400 hover:text-surface-200'}"
				onclick={() => (activeTab = 'clusters')}
			>
				{$_('results.tabs.clusters')}
				{#if clusters.length > 0}
					<span class="ml-1.5 text-xs opacity-75">({clusters.length})</span>
				{/if}
			</button>
		</div>

		<!-- Recommended Tab -->
		{#if activeTab === 'recommended'}
			<div class="flex flex-col gap-3">
				{#if topRecommended.length === 0 && loading}
					<div class="card preset-tonal-surface p-4 flex items-center gap-3 animate-pulse">
						<span class="size-2 rounded-full bg-primary-500 animate-ping"></span>
						<p class="text-surface-400 text-sm">{$_('results.recommended.computing')}</p>
					</div>
				{:else if topRecommended.length === 0 && done}
					<div class="card preset-tonal-surface p-6 text-center text-surface-500 text-sm">
						{$_('results.recommended.noneFound')}
					</div>
				{:else}
					<p class="text-xs text-surface-500">
						{$_('results.recommended.topN', { values: { count: topRecommended.length } })}
					</p>
					{#each topRecommended as repo, i (repo.name)}
						{@render RepoCard(repo, i + 1)}
					{/each}
					{#if loading}
						<div class="card preset-tonal-surface p-3 flex items-center gap-3 animate-pulse">
							<span class="size-2 rounded-full bg-primary-500 animate-ping"></span>
							<p class="text-surface-400 text-sm">{$_('results.recommended.stillComputing')}</p>
						</div>
					{/if}
				{/if}
			</div>
		{/if}

		<!-- Clusters Tab -->
		{#if activeTab === 'clusters'}
			<div class="flex flex-col gap-5">
				{#if clusters.length === 0 && loading}
					<div class="card preset-tonal-surface p-4 flex items-center gap-3 animate-pulse">
						<span class="size-2 rounded-full bg-primary-500 animate-ping"></span>
						<p class="text-surface-400 text-sm">{$_('results.clusters.computing')}</p>
					</div>
				{:else if clusters.length === 0 && done}
					<div class="card preset-tonal-surface p-6 text-center text-surface-500 text-sm">
						{$_('results.clusters.noData')}
					</div>
				{:else}
					<p class="text-xs text-surface-500">
						{$_('results.clusters.hiddenInfo')}
					</p>
					{#each clusterGroups as group}
						<div class="flex flex-col gap-3">
							<!-- Cluster header -->
							<div class="card preset-tonal-warning p-3">
								<div class="flex items-center gap-2 mb-2">
									<svg xmlns="http://www.w3.org/2000/svg" class="size-4 text-warning-400 shrink-0" viewBox="0 0 24 24" fill="currentColor">
										<path d="M21 6.375c0 2.692-4.03 4.875-9 4.875S3 9.067 3 6.375 7.03 1.5 12 1.5s9 2.183 9 4.875z"/>
										<path d="M12 12.75c2.685 0 5.19-.586 7.078-1.609a8.283 8.283 0 001.897-1.384c.016.121.025.244.025.368C21 12.817 16.97 15 12 15s-9-2.183-9-4.875c0-.124.009-.247.025-.368a8.285 8.285 0 001.897 1.384C6.809 12.164 9.315 12.75 12 12.75z"/>
										<path d="M12 16.5c2.685 0 5.19-.586 7.078-1.609a8.282 8.282 0 001.897-1.384c.016.121.025.244.025.368 0 2.692-4.03 4.875-9 4.875s-9-2.183-9-4.875c0-.124.009-.247.025-.368a8.284 8.284 0 001.897 1.384C6.809 15.914 9.315 16.5 12 16.5z"/>
									</svg>
									<p class="text-sm font-semibold text-warning-300">
										{$_('results.clusters.clusterHeader', { values: { id: group.cluster?.cluster_id, total: group.cluster?.cluster_total } })}
										<span class="font-normal text-warning-400 ml-1">
											({$_('results.clusters.shown', { values: { count: group.repos.length } })}
											{#if group.hiddenCount > 0}, {$_('results.clusters.hiddenBelow', { values: { count: group.hiddenCount } })}{/if})
										</span>
									</p>
								</div>

								{#if group.cluster?.top_tags && group.cluster.top_tags.length > 0}
									<div class="flex flex-wrap gap-1 mb-2">
										{#each group.cluster.top_tags as tag}
											<span class="badge preset-tonal-warning text-xs px-2 py-0.5">{tag}</span>
										{/each}
									</div>
								{/if}

								{#if group.cluster?.member_names && group.cluster.member_names.length > 0}
									<div class="border-t border-warning-700/30 pt-2">
										<p class="text-xs text-warning-500 mb-1.5 font-medium">{$_('results.clusters.yourRepos')}</p>
										<div class="flex flex-wrap gap-1">
											{#each group.cluster.member_names as name}
												<a
													href="https://github.com/{name}"
													target="_blank"
													rel="noopener noreferrer"
													class="text-xs bg-warning-900/40 hover:bg-warning-800/50 text-warning-300 px-2 py-0.5 rounded-md font-mono transition-colors"
												>
													{name}
												</a>
											{/each}
										</div>
									</div>
								{/if}
							</div>

							{#if group.repos.length > 0}
								{#each group.repos as repo, i (repo.name)}
									{@render RepoCard(repo, i + 1)}
								{/each}
							{:else}
								<p class="text-xs text-surface-500 italic pl-2">
									{$_('results.clusters.noneAboveThreshold')}
								</p>
							{/if}
						</div>
					{/each}

					{#if loading}
						<div class="card preset-tonal-surface p-3 flex items-center gap-3 animate-pulse">
							<span class="size-2 rounded-full bg-primary-500 animate-ping"></span>
							<p class="text-surface-400 text-sm">{$_('results.clusters.moreStreaming')}</p>
						</div>
					{/if}
				{/if}
			</div>
		{/if}
	{/if}
</div>

{#snippet RepoCard(repo: RepoResult, rank: number)}
	<a
		href="https://github.com/{repo.name}"
		target="_blank"
		rel="noopener noreferrer"
		class="card preset-tonal-surface hover:preset-outlined-primary-500 p-4 flex flex-col gap-3 transition-all no-underline group"
	>
		<div class="flex items-start justify-between gap-3">
			<div class="flex items-center gap-2 min-w-0">
				<span class="text-surface-500 text-xs font-mono shrink-0 w-5 text-right">#{rank}</span>
				<p class="font-semibold text-sm group-hover:text-primary-400 transition-colors truncate">
					{repo.name}
				</p>
			</div>
			<div class="flex items-center gap-2 shrink-0">
				{#if repo.pct_rank !== null}
					<span class="badge preset-tonal-success text-xs px-2">{pctLabel(repo.pct_rank)}</span>
				{/if}
				<svg xmlns="http://www.w3.org/2000/svg" class="size-4 text-surface-500 group-hover:text-primary-400 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
				</svg>
			</div>
		</div>

		<div class="flex items-center gap-3">
			<div class="flex-1 h-1.5 bg-surface-700 rounded-full overflow-hidden">
				<div
					class="h-full rounded-full transition-all {scoreColor(repo.score)}"
					style="width: {scoreBar(repo.score)}%"
				></div>
			</div>
			<div class="text-xs font-mono text-surface-400 shrink-0 w-28 text-right">
				{repo.score.toFixed(4)}
				{#if repo.calibrated !== null}
					<span class="text-surface-600 ml-1">cal: {repo.calibrated.toFixed(2)}</span>
				{/if}
			</div>
		</div>

		{#if repo.matching_tags.length > 0}
			<div class="flex flex-wrap gap-1">
				{#each repo.matching_tags as tag}
					<span class="badge preset-tonal-primary text-xs px-2 py-0.5">{tag}</span>
				{/each}
			</div>
		{/if}
	</a>
{/snippet}
