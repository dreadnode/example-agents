<?php

/**
 * Extended Taxonomy Terms Settings Module
 * Settings > Widget Options :: Extended Taxonomy Terms
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Create Card Module for Extended Taxonomy Terms Options
 *
 * @since 3.0
 * @return void
 */
if (!function_exists('widgetopts_settings_taxonomies')) :
	function widgetopts_settings_taxonomies()
	{ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="taxonomies">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . 'features/post-types-taxonomies-widget-visibility/'); ?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e('Extended Taxonomy Terms', 'widget-options'); ?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e('Extend each widget visibility for custom post types taxonomies and terms.', 'widget-options'); ?>
				</p>
			</div>
		</li>
<?php
	}
// add_action('widgetopts_module_cards', 'widgetopts_settings_taxonomies', 140);
endif;
?>