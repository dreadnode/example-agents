<?php

/**
 * Links Settings Module
 * Settings > Widget Options :: Link Widget
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Create Card Module for Custom Links Options
 *
 * @since 3.0
 * @return void
 */
if (!function_exists('widgetopts_settings_urls')) :
	function widgetopts_settings_urls()
	{ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="urls">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . 'features/target-url-wildcards/'); ?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e('URL & Wildcards Restrictions', 'widget-options'); ?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e('Show or hide widgets by URL and/or use <code>*</code> to create a URL wildcard restrictions.', 'widget-options'); ?>
				</p>
			</div>
		</li>
<?php
	}
// add_action( 'widgetopts_module_cards', 'widgetopts_settings_urls', 190 );
endif;
?>