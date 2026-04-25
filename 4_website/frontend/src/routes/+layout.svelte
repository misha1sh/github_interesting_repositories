<script lang="ts">
	import { isLoading } from 'svelte-i18n';
	import { setupI18n } from '$lib/i18n/index';
	import './+layout.css';

	let { children } = $props();

	setupI18n();
</script>

{#if !$isLoading}
	{@render children()}
{/if}
