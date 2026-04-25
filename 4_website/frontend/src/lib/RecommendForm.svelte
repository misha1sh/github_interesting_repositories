<script lang="ts">
	import { _ } from 'svelte-i18n';

	export type FormParams = {
		username: string;
		top_k: number;
		model: 'nn' | 'simple';
		include_forked: boolean;
		include_user_repos: boolean;
		clusters: number;
		clustering_type: 'k-means' | 'agglomerative' | 'gmm' | 'dbscan';
	};

	type Props = {
		onSubmit: (params: FormParams) => void;
		loading: boolean;
	};

	let { onSubmit, loading }: Props = $props();

	let username = $state('');
	let top_k = $state(30);
	let model = $state<'nn' | 'simple'>('nn');
	let include_forked = $state(true);
	let include_user_repos = $state(true);
	let clusters = $state(0);
	let clustering_type = $state<'k-means' | 'agglomerative' | 'gmm' | 'dbscan'>('k-means');

	let usernameError = $state('');

	function handleSubmit(e: Event) {
		e.preventDefault();
		usernameError = '';
		if (!username.trim()) {
			usernameError = $_('form.usernameRequired');
			return;
		}
		onSubmit({ username: username.trim(), top_k, model, include_forked, include_user_repos, clusters, clustering_type });
	}
</script>

<form onsubmit={handleSubmit} class="flex flex-col gap-5">
	<!-- Username -->
	<div class="flex flex-col gap-1">
		<label for="username" class="label font-semibold text-sm">
			{$_('form.username')} <span class="text-error-500">*</span>
		</label>
		<div class="input-group grid-cols-[auto_1fr]">
			<div class="ig-cell preset-tonal">
				<svg xmlns="http://www.w3.org/2000/svg" class="size-4" viewBox="0 0 24 24" fill="currentColor">
					<path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.866-.013-1.7-2.782.603-3.369-1.342-3.369-1.342-.454-1.154-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.091-.646.35-1.087.636-1.337-2.22-.252-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0 1 12 6.836a9.58 9.58 0 0 1 2.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.202 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.744 0 .267.18.579.688.481C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
				</svg>
			</div>
			<input
				id="username"
				type="text"
				class="ig-input"
				placeholder={$_('form.usernamePlaceholder')}
				bind:value={username}
				disabled={loading}
				autocomplete="off"
				spellcheck="false"
			/>
		</div>
		{#if usernameError}
			<p class="text-error-500 text-xs mt-1">{usernameError}</p>
		{/if}
	</div>

	<!-- Model type -->
	<div class="flex flex-col gap-2">
		<span class="label font-semibold text-sm">{$_('form.model')}</span>
		<div class="flex gap-2">
			{#each [{ v: 'nn', key: 'form.modelNn' }, { v: 'simple', key: 'form.modelSimple' }] as opt}
				<button
					type="button"
					class="btn btn-sm {model === opt.v ? 'preset-filled-primary-500' : 'preset-outlined-surface-200-800'}"
					onclick={() => (model = opt.v as 'nn' | 'simple')}
					disabled={loading}
				>
					{$_(opt.key)}
				</button>
			{/each}
		</div>
	</div>

	<!-- Top K -->
	<div class="flex flex-col gap-2">
		<label for="top_k" class="label font-semibold text-sm">
			{$_('form.topK')} <span class="text-primary-500 font-bold">{top_k}</span>
		</label>
		<input
			id="top_k"
			type="range"
			class="input"
			min="5"
			max="100"
			step="5"
			bind:value={top_k}
			disabled={loading}
		/>
		<div class="flex justify-between text-xs text-surface-400">
			<span>5</span><span>100</span>
		</div>
	</div>

	{#if model === 'nn'}
		<!-- Clustering type -->
		<div class="flex flex-col gap-2">
			<label for="clustering_type" class="label font-semibold text-sm">{$_('form.clusteringAlgorithm')}</label>
			<select id="clustering_type" class="select" bind:value={clustering_type} disabled={loading}>
				<option value="k-means">{$_('form.clusteringKMeans')}</option>
				<option value="agglomerative">{$_('form.clusteringAgglomerative')}</option>
				<option value="gmm">{$_('form.clusteringGmm')}</option>
				<option value="dbscan">{$_('form.clusteringDbscan')}</option>
			</select>
		</div>

		<!-- Clusters -->
		<div class="flex flex-col gap-2">
			<label for="clusters" class="label font-semibold text-sm">
				{$_('form.interestClusters')} <span class="text-primary-500 font-bold">{clusters === 0 ? $_('form.clustersAuto') : clusters}</span>
			</label>
			<input
				id="clusters"
				type="range"
				class="input"
				min="0"
				max="10"
				step="1"
				bind:value={clusters}
				disabled={loading}
			/>
			<div class="flex justify-between text-xs text-surface-400">
				<span>{$_('form.clustersAuto')}</span><span>10</span>
			</div>
		</div>
	{/if}

	<!-- Toggles -->
	<div class="card preset-tonal-surface p-4 flex flex-col gap-4">
		<p class="text-sm font-semibold text-surface-500 uppercase tracking-wide">{$_('form.options')}</p>
		<label class="flex items-center justify-between gap-4 cursor-pointer">
			<span class="text-sm">
				{$_('form.includeForked')}
				<span class="text-xs text-surface-400 block">{$_('form.includeForkedDesc')}</span>
			</span>
			<input type="checkbox" class="checkbox" bind:checked={include_forked} disabled={loading} />
		</label>
		<hr class="hr" />
		<label class="flex items-center justify-between gap-4 cursor-pointer">
			<span class="text-sm">
				{$_('form.includeOwn')}
				<span class="text-xs text-surface-400 block">{$_('form.includeOwnDesc')}</span>
			</span>
			<input type="checkbox" class="checkbox" bind:checked={include_user_repos} disabled={loading} />
		</label>
	</div>

	<!-- Submit -->
	<button
		type="submit"
		class="btn preset-filled-primary-500 w-full font-bold text-base"
		disabled={loading}
	>
		{#if loading}
			<span class="animate-spin mr-2 inline-block size-4 border-2 border-white border-t-transparent rounded-full"></span>
			{$_('form.recommending')}
		{:else}
			{$_('form.findRepos')}
		{/if}
	</button>
</form>
