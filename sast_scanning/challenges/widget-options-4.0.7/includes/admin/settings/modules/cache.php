<?php
/**
 * Widget Caching Settings Module
 * Settings > Widget Options :: Cache
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       3.2
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Fixed Widget Options
 *
 * @since 3.2
 * @return void
 */
if( !function_exists( 'widgetopts_settings_cache' ) ):
	function widgetopts_settings_cache(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="cache">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/wordpress-widget-cache/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Widget Cache', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Improve loading and performance by caching widget output using Transient API.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_cache', 175 );
endif;
?>
