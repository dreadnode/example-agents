<?php
/**
 * Columns Settings Module
 * Settings > Widget Options :: Column Display
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Column Display Options
 *
 * @since 3.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_columns' ) ):
	function widgetopts_settings_columns(){
		?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="columns">
				<div class="widgetopts-module-card-content">
					<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/responsive-wordpress-widget-columns/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
					<h2><?php _e( 'Column Display', 'widget-options' );?></h2>
					<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
					<p class="widgetopts-module-desc">
						<?php _e( 'Manage your widgets display as columns, set different columns for specific devices.', 'widget-options' );?>
					</p>
				</div>
			</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_columns', 90 );
endif;
?>
