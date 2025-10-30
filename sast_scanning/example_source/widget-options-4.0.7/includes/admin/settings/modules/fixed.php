<?php

/**
 * Fixed Widget Settings Module
 * Settings > Widget Options :: Fixed Widget
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Create Card Module for Fixed Widget Options
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */
if (!function_exists('widgetopts_settings_fixed')) :
	function widgetopts_settings_fixed()
	{ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="fixed">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . 'features/sticky-fixed-wordpress-widgets/'); ?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e('Fixed Widget', 'widget-options'); ?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e('Add fixed positioning to each widget when the page is scrolled.', 'widget-options'); ?>
				</p>
			</div>
		</li>
<?php
	}
	add_action('widgetopts_module_cards', 'widgetopts_settings_fixed', 69);
endif;
?>