<?php
/**
 * Disable Widgets Settings Module
 * Settings > Widget Options :: Disable Widgets
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Disable Widgets Options
 *
 * @since 3.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_disable_widgets' ) ):
	function widgetopts_settings_disable_widgets(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="disable_widgets">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Disable Widgets', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Disable several widgets that you won\'t be using to lessen widget dashboard space.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_disable_widgets', 150 );
endif;
?>
