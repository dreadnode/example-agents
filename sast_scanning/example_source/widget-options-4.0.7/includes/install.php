<?php

/**
 * Install Function
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
 */
// Exit if accessed directly
if (!defined('ABSPATH')) exit;

//check if free version is activated
if (!function_exists('widgetopts_upgraded')) {
	add_action('admin_notices', 'widgetopts_upgraded');
	function widgetopts_upgraded()
	{
		if (is_plugin_active('widget-options/plugin.php') && is_plugin_active('extended-widget-options/plugin.php')) { ?>
			<div class="widgetopts_activated_notice notice-error notice" style="box-shadow: 0 1px 1px 0 rgba(0,0,0,.1);">
				<p>
					<?php _e('Please deactivate <strong>Widget Options</strong> Plugin, it may cause issue with the extended plugin version. Thanks!', 'widget-options'); ?>
				</p>
			</div>
<?php }
	}
}

//add settings link on plugin page
if (!function_exists('widgetopts_filter_plugin_actions')) {
	add_action('plugin_action_links_' . plugin_basename(WIDGETOPTS_PLUGIN_FILE), 'widgetopts_filter_plugin_actions');
	function widgetopts_filter_plugin_actions($links)
	{

		if (!is_array($links)) {
			$links = array();
		}

		$links[]  = '<a href="' . esc_url(admin_url('options-general.php?page=widgetopts_plugin_settings')) . '">' . __('Settings', 'widget-options') . '</a>';
		$upgrade_link = apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . 'pricing/?utm_source=upgradebtn&utm_medium=plugins&utm_campaign=widgetoptspluginlink');
		$links[]  = '<a href="' . esc_url($upgrade_link) . '" target="_blank" style="color: #3db634">' . __('Upgrade', 'widget-options') . '</a>';
		return $links;
	}
}

//register default values
if (!function_exists('widgetopts_register_defaults')) {
	register_activation_hook(WIDGETOPTS_PLUGIN_FILE, function () {
		add_option('Activated_Plugin', WIDGETOPTS_PLUGIN_FILE);
		widgetopts_register_defaults();
	});

	add_action('admin_init', function () {
		if (is_admin() && get_option('Activated_Plugin') == WIDGETOPTS_PLUGIN_FILE) {
			delete_option('Activated_Plugin');
			exit(wp_redirect(admin_url('options-general.php?page=widgetopts_plugin_settings')));
		}
	});

	add_action('plugins_loaded', 'widgetopts_register_defaults');
	function widgetopts_register_defaults()
	{
		if (is_admin()) {

			if (!get_option('widgetopts_installDate')) {
				add_option('widgetopts_installDate', date('Y-m-d h:i:s'));
			}

			if (!get_option('_widgetopts_default_registered_')) {
				//activate free version modules
				add_option('widgetopts_tabmodule-visibility', 'activate');
				add_option('widgetopts_tabmodule-devices', 'activate');
				add_option('widgetopts_tabmodule-alignment', 'activate');
				add_option('widgetopts_tabmodule-hide_title', 'activate');
				add_option('widgetopts_tabmodule-classes', 'activate');
				add_option('widgetopts_tabmodule-logic', 'activate');
				add_option('widgetopts_tabmodule-state', 'activate');
				add_option('widgetopts_tabmodule-classic_widgets_screen', 'activate');
				//add free version settings
				$defaults = array(
					'visibility' 	=> 	array(
						'post_type'		=> '1',
						'taxonomies'	=> '1',
						'misc'			=> '1'
					),
					'classes' 		=> 	array(
						'id'			=> '1',
						'type'			=> 'both'
					),
				);
				//upgraded settings from previous version
				$options    = get_option('extwopts_class_settings');
				if (isset($options['class_field'])) {
					$defaults['classes']['type'] = $options['class_field'];
				}
				if (isset($options['classlists'])) {
					$defaults['classes']['classlists'] = $options['classlists'];
				}
				add_option('widgetopts_tabmodule-settings', serialize($defaults));
				add_option('_widgetopts_default_registered_', '1');
				delete_transient('widgetopts_tabs_transient'); //remove transient for settings
				delete_option('widgetopts_settings');
			}

			//make sure to delete previous pages cache
			if (!get_option('widgetopts_removed_global_pages')) {
				delete_option('widgetopts_global_pages');
				add_option('widgetopts_removed_global_pages', 1);
			}
		}
	}
}

?>