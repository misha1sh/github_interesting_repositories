import { addMessages, init, locale, getLocaleFromNavigator } from 'svelte-i18n';
import { get } from 'svelte/store';
import en from './en.json';
import ru from './ru.json';

const STORAGE_KEY = 'locale';
const SUPPORTED = ['en', 'ru'];

addMessages('en', en);
addMessages('ru', ru);

export function setupI18n() {
	const saved = typeof localStorage !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
	const browser = getLocaleFromNavigator()?.slice(0, 2) ?? 'en';
	const detected = saved && SUPPORTED.includes(saved) ? saved : (SUPPORTED.includes(browser) ? browser : 'en');

	init({
		fallbackLocale: 'en',
		initialLocale: detected,
	});
}

export function toggleLocale() {
	const current = get(locale)?.slice(0, 2) ?? 'en';
	const next = current === 'ru' ? 'en' : 'ru';
	locale.set(next);
	if (typeof localStorage !== 'undefined') {
		localStorage.setItem(STORAGE_KEY, next);
	}
}

export { locale };
